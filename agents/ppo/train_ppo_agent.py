import argparse
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
from rlcard.utils import set_seed

from ppo_agent import PPOAgent
from ppo_value_estimator import ValueEstimator

ENV_ID = 'no-limit-holdem'
AGENT_ID = 0

def compute_value(value_estimator, state):
    win_prob = value_estimator.calculate_heuristic_win_prob(state['obs'].tolist())
    my_chips = state['raw_obs']['my_chips']  
    other_chips = state['raw_obs']['pot'] - my_chips
    return other_chips * win_prob - my_chips * (1 - win_prob)


def agent_step(env, value_estimator, old_agent, new_agent):
    """
    Steps agent once, keeping track of [advantage], [old_prob], and [new_prob].
    Must be called when [env] has [new_agent] at [AGENT_ID].
    """
    prev_state = env.get_state(AGENT_ID)
    
    _, old_action_probs = old_agent.step_with_probs(prev_state, no_grad=True)
    new_action, new_action_probs = new_agent.step_with_probs(prev_state)

    env.step(new_action)

    old_prob = old_action_probs[new_action.value]
    new_prob = new_action_probs[new_action.value]

    return new_prob, old_prob


def rollout(env, value_estimator, old_agent):
    """
    Rollouts agents for one round.
    """ 
    advantages = []
    old_probs = []
    new_probs = []

    old_state = None
    while not env.is_over():
        player_id = env.get_player_id()
        agent = env.agents[player_id]
        state = env.get_state(player_id)

        if player_id == AGENT_ID:
            if old_state != None:
                old_value = compute_value(value_estimator, old_state)
                new_value = compute_value(value_estimator, state)
                advantages.append(new_value - old_value)

            new_prob, old_prob = agent_step(env, value_estimator, old_agent, agent)
            old_probs.append(old_prob)
            new_probs.append(new_prob)

            old_state = copy.deepcopy(state)
        else:
            state = env.get_state(player_id)
            action = agent.step(state)
            env.step(action)

    return advantages, old_probs[:-1], new_probs[:-1]


def rollout_all_actors(env, value_estimator, old_agent, num_actors):
    all_advantages = []
    all_old_probs = []
    all_new_probs = []

    for _ in range(num_actors):
        # TODO: May have to make multiple environments to run in parallel.
        env.reset() 
        advantages, old_probs, new_probs = rollout(env, value_estimator, old_agent)
        all_advantages.extend(advantages)
        all_old_probs.extend(old_probs)
        all_new_probs.extend(new_probs)

    return all_advantages, all_old_probs, all_new_probs


LOG_EPSILON = 10**-7
def train(env, 
          agent, 
          num_iters, 
          num_actors, 
          clip_epsilon, 
          lr, 
          checkpoint_folder,
          checkpoint_freq,
          checkpoint_folder_id):
    old_agent = agent
    value_estimator = ValueEstimator()
    optimizer = optim.Adam(agent.policy.parameters(), lr=lr)

    losses = []
    for i in tqdm(range(num_iters)):
        advantages, old_probs, new_probs = rollout_all_actors(env, value_estimator, old_agent, num_actors)
        if len(advantages) == 0:
            continue

        old_log_probs = torch.log(torch.stack(old_probs) + LOG_EPSILON)
        new_log_probs = torch.log(torch.stack(new_probs) + LOG_EPSILON)
        prob_ratios = torch.exp(new_log_probs - old_log_probs)

        advantages = torch.tensor(advantages)
        surrogate_loss = agent.policy.compute_surrogate_loss(prob_ratios, advantages, clip_epsilon)
        avg_surrogate_loss = torch.mean(surrogate_loss) 
        losses.append(avg_surrogate_loss.item())

        old_agent = copy.deepcopy(agent)

        optimizer.zero_grad()
        avg_surrogate_loss.backward()
        optimizer.step()

        if i % checkpoint_freq == 0 and i > 0:
            checkpoint_path = f'{checkpoint_folder}/{checkpoint_folder_id}/model_{i}.pt'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) 
            torch.save(agent, checkpoint_path)
            
    plt.plot(losses)
    plt.show()


def env_shape(env):
    state_channels = env.state_shape[0][0]
    action_channels = env.num_actions
    return state_channels, action_channels


def init_env(env, agent, num_random_agents):
    """
    Initializes the poker environment with the specified agent and a number of 
    random agents. The specified agent will always be the first agent. 
    """
    random_agents = [RandomAgent(num_actions=env.num_actions) 
                     for _ in range(num_random_agents)]
    env.set_agents([agent, *random_agents])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(f"Train PPO agent on {ENV_ID}")
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--num_random_agents', type=int, default=1) 
    parser.add_argument('--num_iters', type=int, default=200)
    parser.add_argument('--num_actors', type=int, default=50)
    parser.add_argument('--clip_epsilon', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--checkpoint_folder', type=str, default='models/ppo')
    parser.add_argument('--checkpoint_freq', type=int, default=500)
    args = parser.parse_args()

    set_seed(42)
    checkpoint_folder_id = current_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

    env = rlcard.make(ENV_ID)
    state_channels, action_channels = env_shape(env)
    agent = PPOAgent(state_channels, args.hidden_dim, action_channels)
    init_env(env, agent, args.num_random_agents)

    train(env, 
          agent, 
          args.num_iters, 
          args.num_actors, 
          args.clip_epsilon, 
          args.lr, 
          args.checkpoint_folder,
          args.checkpoint_freq,
          checkpoint_folder_id)