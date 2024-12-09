import argparse
import random
import os
import copy
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, get_device
from rlcard.games.nolimitholdem.round import Action

from ppo_agent import PPOAgent
from ppo_value_estimator import ValueEstimator

from loader import load_model

ENV_ID = 'no-limit-holdem'
AGENT_ID = 0

def step_only_legal(env, action, state):
    if action in state['raw_legal_actions']:
        env.step(action)
    else:
        env.step(random.choice(state['raw_legal_actions']))


def compute_value(value_estimator, state):
    win_prob = value_estimator.calculate_heuristic_win_prob(state['obs'].tolist())
    my_chips = state['raw_obs']['my_chips']  
    other_chips = state['raw_obs']['pot'] - my_chips
    return other_chips * win_prob - my_chips * (1 - win_prob)


NON_LEGAL_ADVANTAGE = -3000
def rollout_action(env, value_estimator, old_agent, num_action_samples):
    """
    Rollouts agent for one action, taking the expected TD value over samples.
    """
    start_state = copy.deepcopy(env.get_state(AGENT_ID))
    new_agent = env.agents[AGENT_ID]

    _, old_action_probs = old_agent.step_with_probs(start_state, no_grad=True)
    new_action, new_action_probs = new_agent.step_with_probs(start_state)

    old_prob = old_action_probs[new_action.value]
    new_prob = new_action_probs[new_action.value]

    if Action(new_action) not in start_state['raw_legal_actions']:
        env.step(random.choice(start_state['raw_legal_actions']))
        return old_prob, new_prob, new_action_probs, NON_LEGAL_ADVANTAGE

    total_advantage = 0
    env.step(new_action)
    for _ in range(num_action_samples):
        start_value = compute_value(value_estimator, start_state)

        num_steps = 0 
        while not env.is_over() and env.get_player_id() != AGENT_ID:
            other_agent_id = env.get_player_id()
            other_agent = env.agents[other_agent_id]
            other_agent_state = env.get_state(other_agent_id)
            other_agent_action = other_agent.step(other_agent_state)
            step_only_legal(env, other_agent_action, other_agent_state)
            num_steps += 1

        if env.is_over():
            end_value = env.get_payoffs()[AGENT_ID]
        else:
            end_state = env.get_state(AGENT_ID)
            end_value = compute_value(value_estimator, end_state)
        total_advantage += end_value - start_value

        for _ in range(num_steps):
            env.step_back()

    mean_advantage = total_advantage / num_action_samples
    return old_prob, new_prob, new_action_probs, mean_advantage


def rollout(env, value_estimator, old_agent, num_action_samples):
    """
    Rollouts agents for one round.
    """ 
    advantages = []
    old_probs = []
    new_probs = []
    all_action_probs = []

    while not env.is_over():
        player_id = env.get_player_id()
        agent = env.agents[player_id]
        state = env.get_state(player_id)

        if player_id == AGENT_ID:
            old_prob, new_prob, action_probs, advantage = rollout_action(env, value_estimator, old_agent, num_action_samples)
            old_probs.append(old_prob)
            new_probs.append(new_prob)
            advantages.append(advantage)
            all_action_probs.append(action_probs)
        else:
            state = env.get_state(player_id)
            action = agent.step(state)
            step_only_legal(env, action, state)

    payoff = env.get_payoffs()[AGENT_ID]

    return advantages, old_probs, new_probs, all_action_probs, payoff


def rollout_all_actors(env, value_estimator, old_agent, num_actors, num_action_samples):
    all_advantages = []
    all_old_probs = []
    all_new_probs = []
    all_action_probs = []
    payoff = 0

    for _ in range(num_actors):
        # TODO: May have to make multiple environments to run in parallel.
        env.reset() 
        advantages, old_probs, new_probs, action_probs, single_payoff = rollout(env, value_estimator, old_agent, num_action_samples)
        all_advantages.extend(advantages)
        all_old_probs.extend(old_probs)
        all_new_probs.extend(new_probs)
        all_action_probs.extend(action_probs)
        payoff += single_payoff

    return all_advantages, all_old_probs, all_new_probs, all_action_probs, payoff


LOG_EPSILON = 10**-7
def train(env, 
          agent, 
          num_iters, 
          num_actors, 
          num_action_samples,
          clip_epsilon, 
          lr, 
          beta,
          self_play,
          checkpoint_folder,
          checkpoint_freq,
          checkpoint_folder_id):
    old_agent = agent
    value_estimator = ValueEstimator()
    optimizer = optim.Adam(agent.policy.parameters(), lr=lr)

    losses = []
    cumulative_payoff = []
    for i in tqdm(range(num_iters)):
        advantages, old_probs, new_probs, action_probs, payoff = rollout_all_actors(env, value_estimator, old_agent, num_actors, num_action_samples)
        if len(advantages) == 0:
            continue

        old_log_probs = torch.log(torch.stack(old_probs) + LOG_EPSILON)
        new_log_probs = torch.log(torch.stack(new_probs) + LOG_EPSILON)
        prob_ratios = torch.exp(new_log_probs - old_log_probs)

        advantages = torch.tensor(advantages)
        surrogate_loss = agent.policy.compute_surrogate_loss(prob_ratios, advantages, clip_epsilon)

        action_probs = torch.stack(action_probs)
        entropy = torch.sum(action_probs * torch.log(action_probs + LOG_EPSILON), dim=-1)
        entropy.reshape(surrogate_loss.shape)
        
        total_loss = surrogate_loss - beta * entropy
        mean_loss = torch.mean(total_loss) 
        losses.append(surrogate_loss.mean().item())

        old_agent = copy.deepcopy(agent)

        if self_play:
            env.agents = [agent, old_agent, *env.agents[1:-1]]

        optimizer.zero_grad()
        mean_loss.backward()
        optimizer.step()

        cumulative_payoff.append(payoff + (cumulative_payoff[-1] if cumulative_payoff else 0))

        if (i + 1) % checkpoint_freq == 0:
            base_path = f'{checkpoint_folder}/{checkpoint_folder_id}'
            checkpoint_path = f'{base_path}/checkpoints/model_{i + 1}.pt'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) 
            torch.save(agent, checkpoint_path)

            loss_graph_path = f'{base_path}/loss.png'
            plt.plot(losses)
            plt.yscale('log')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.savefig(loss_graph_path)
            plt.close()

            payoff_graph_path = f'{base_path}/payoff.png'
            plt.plot(cumulative_payoff)
            plt.xlabel('Iterations')
            plt.ylabel('Cumulative Payoff')
            plt.savefig(payoff_graph_path)
            plt.close()


def env_shape(env):
    state_channels = env.state_shape[0][0]
    action_channels = env.num_actions
    return state_channels, action_channels


def init_env(env, agent, num_other_agents, self_play, play_against):
    """
    Initializes the poker environment with the specified agent and a number of 
    other agents. The specified agent will always be the first agent. 
    """
    if self_play:
        other_agents = [copy.deepcopy(agent) 
                        for _ in range(num_other_agents)]
    elif play_against:
        other_agents = [load_model(play_against, env, i + 1, get_device())
                        for i in range(num_other_agents)]
    else:
        other_agents = [RandomAgent(num_actions=env.num_actions) 
                        for _ in range(num_other_agents)]

    env.set_agents([agent, *other_agents])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(f"Train PPO agent on {ENV_ID}")
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_other_agents', type=int, default=1) 
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--num_actors', type=int, default=1)
    parser.add_argument('--num_action_samples', type=int, default=50)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--checkpoint_folder', type=str, default='models/ppo')
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    parser.add_argument('--self_play', action='store_true') 
    parser.add_argument('--play_against', type=str, default='')
    args = parser.parse_args()

    if args.self_play and args.play_against:
        raise Exception('Must choose one of --self_play and --play_against.')

    set_seed(42)
    checkpoint_folder_id = current_date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") 

    env = rlcard.make(ENV_ID, { 'allow_step_back' : True })
    state_channels, action_channels = env_shape(env)
    agent = PPOAgent(state_channels, args.hidden_dim, action_channels)
    init_env(env, agent, args.num_other_agents, self_play=args.self_play, play_against=args.play_against)

    train(env, 
          agent, 
          args.num_iters, 
          args.num_actors, 
          args.num_action_samples,
          args.clip_epsilon, 
          args.lr, 
          args.beta,
          args.self_play,
          args.checkpoint_folder,
          args.checkpoint_freq,
          checkpoint_folder_id)