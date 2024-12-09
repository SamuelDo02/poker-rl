import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import rlcard
from rlcard.agents import RandomAgent

from ppo_agent import PPOAgent
from ppo_value_estimator import ValueEstimator

ENV_ID = 'no-limit-holdem'
AGENT_IDX = 0

def rollout(env):
    """
    Rollouts a policy for one round.
    """
    trajectories, payoffs = env.run(is_training=True)
    print(payoffs)


def rollout_all_actors(env, num_actors):
    for _ in range(num_actors):
        rollout(env) 


def train(env, agent, num_iters, num_actors, epsilon):
    for _ in tqdm(range(num_iters)):
        rollout_all_actors(env, num_actors)
        old_policy = copy(policy) # TODO: Can we do this without copying?
        # make a new dim for num_actors to do rollouts in parallel
        actors_states = states.reshape([states.shape[0], num_actors, 1])
        # rollout current policy in environment for T timesteps
        cur_log_probs, prev_log_probs, advantages, states = rollout(actors_states)
        ratios = torch.exp(cur_log_probs - prev_log_probs)
        surrogate_loss = policy.compute_surrogate_loss(ratios, advantages, epsilon)
        for k in self.num_epochs:
            surrogate_loss.backward()


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
    env.set_agents(random_agents)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(f"Train PPO agent on {ENV_ID}")
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--num_random_agents', type=int, default=1) 
    parser.add_argument('--num_iters', type=int, default=200)
    parser.add_argument('--num_actors', type=int, default=5)
    parser.add_argument('--rollout_length', type=int, default=5)
    parser.add_argument('--epsilon', type=int, default=0.2)
    args = parser.parse_args()

    env = rlcard.make(ENV_ID)
    state_channels, action_channels = env_shape(env)
    agent = PPOAgent(state_channels, args.hidden_dim, action_channels)
    init_env(env, agent, args.num_random_agents)

    train(env, agent, args.num_iters, args.num_actors, args.epsilon)