import argparse
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

import rlcard
from rlcard.agents import RandomAgent

from ppo_agent import PPOAgent
from ppo_value_estimator import ValueEstimator

ENV_ID = 'no-limit-holdem'
AGENT_ID = 0

def compute_advantage(value_estimator, prev_obs, new_obs):
    prev_val = value_estimator.calculate_heuristic_win_prob(prev_obs.tolist())
    curr_val = value_estimator.calculate_heuristic_win_prob(new_obs.tolist())
    return curr_val - prev_val


def agent_step(env, value_estimator, old_agent, new_agent):
    """
    Steps agent once, keeping track of [advantage], [old_prob], and [new_prob].
    Must be called when [env] has [new_agent] at [AGENT_ID].
    """
    prev_state = env.get_state(AGENT_ID)
    
    _, old_action_probs = old_agent.step_with_probs(prev_state, no_grad=True)
    new_action, new_action_probs = new_agent.step_with_probs(prev_state)

    new_state, _ = env.step(new_action)

    advantage = compute_advantage(value_estimator, prev_state['obs'], new_state['obs'])
    old_prob = old_action_probs[new_action.value]
    new_prob = new_action_probs[new_action.value]

    return advantage, new_prob, old_prob


def rollout(env, value_estimator, old_agent):
    """
    Rollouts agents for one round.
    """ 
    advantages = []
    old_probs = []
    new_probs = []
    
    while not env.is_over():
        player_id = env.get_player_id()
        agent = env.agents[player_id]

        if player_id == AGENT_ID:
            advantage, new_prob, old_prob = agent_step(env, value_estimator, old_agent, agent)
            advantages.append(advantage)
            old_probs.append(old_prob)
            new_probs.append(new_prob)
        else:
            state = env.get_state(player_id)
            action = agent.step(state)
            env.step(action)

    return advantages, old_probs, new_probs


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
def train(env, agent, num_iters, num_actors, clip_epsilon, lr, checkpointpath):
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

        if i % 100 == 0 and i > 0:
            torch.save({
                'model_state_dict': [agent.state_dict()],
                'optimizer_state_dict': [optimizer.state_dict()]
            }, checkpointpath)
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
    parser.add_argument('--num_actors', type=int, default=5)
    parser.add_argument('--rollout_length', type=int, default=5)
    parser.add_argument('--clip_epsilon', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)

    args = parser.parse_args()

    env = rlcard.make(ENV_ID)
    state_channels, action_channels = env_shape(env)
    agent = PPOAgent(state_channels, args.hidden_dim, action_channels)
    init_env(env, agent, args.num_random_agents)

    train(env, agent, args.num_iters, args.num_actors, args.clip_epsilon, args.lr, 'ppo')