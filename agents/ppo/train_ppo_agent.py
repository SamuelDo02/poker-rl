import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import rlcard
from rlcard.agents import RandomAgent

from ppo_agent import PPOAgent

def rollout(self, env):
    """
    Rollouts a policy for one round.
    """
    trajectories, payoffs = game_env.run(is_training=True)


def train(env, agent, num_iters, num_actors, rollout_length, epsilon):
    policy = init_policy()
    for i in tqdm(range(num_iters)):
        old_policy = copy(policy) # TODO: Can we do this without copying?
        # make a new dim for num_actors to do rollouts in parallel
        actors_states = states.reshape([states.shape[0], num_actors, 1])
        # rollout current policy in environment for T timesteps
        cur_log_probs, prev_log_probs, advantages, states = rollout(actors_states, rollout_length)
        ratios = torch.exp(cur_log_probs - prev_log_probs)
        surrogate_loss = policy.compute_surrogate_loss(ratios, advantages, epsilon)
        for k in self.num_epochs:
            surrogate_loss.backward()


def init_env(agent, num_random_agents):
    env = rlcard.make('no-limit-holdem')

    random_agents = [RandomAgent(num_actions=env.num_actions) 
                     for _ in range(num_random_agents)]
    env.set_agents([agent, *random_agents])

    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train PPO agent on No Limit Texas Hold'Em")
    # TODO: Change this default to 1 when we add PPO agent to the list of agents.
    parser.add_argument('--num_random_agents', type=int, default=2) 
    parser.add_argument('--num_iters', type=int, default=200)
    parser.add_argument('--num_actors', type=int, default=5)
    parser.add_argument('--rollout_length', type=int, default=5)
    parser.add_argument('--epsilon', type=int, default=0.2)
    args = parser.parse_args()

    agent = PPOAgent()
    env = init_env(agent, args.num_random_agents)

    train(env, agent, args.num_iters, args.num_actors, args.rollout_length, args.epsilon)