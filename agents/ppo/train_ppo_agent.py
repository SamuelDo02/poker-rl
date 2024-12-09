import argparse
from tqdm import tqdm

import torch
import torch.nn as nn

import rlcard
from rlcard.agents import RandomAgent

def rollout(self, env):
    """
    Rollouts a policy for one round.
    """
    trajectories, payoffs = game_env.run(is_training=True)


def initialize_env(num_random_agents):
    env = rlcard.make('no-limit-holdem')

    # TODO: Add PPO agent to the list.
    random_agents = [RandomAgent(num_actions=env.num_actions) 
                     for _ in range(num_random_agents)]
    env.set_agents(random_agents)

    return env


def train(env, num_iters, num_actors, rollout_length, epsilon):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train PPO agent on No Limit Texas Hold'Em")
    # TODO: Change this default to 1 when we add PPO agent to the list of agents.
    parser.add_argument('--num_random_agents', type=int, default=2) 
    args = parser.parse_args()

    env = initialize_env(args.num_random_agents)
    train(env, num_iters, num_actors, epsilon)