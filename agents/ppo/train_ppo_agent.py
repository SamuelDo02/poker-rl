import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

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
    Must be called when [env ]
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


def rollout_all_actors(env, value_estimator, old_agent, new_agent, num_actors):
    # for _ in range(num_actors):
    env.reset()
    print(rollout(env, value_estimator, old_agent))


def train(env, agent, value_estimator, num_iters, num_actors, epsilon, lr):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    for _ in tqdm(range(num_iters)):
        rollout_all_actors(env, value_estimator, agent, agent, num_actors)
        old_policy = copy(policy) # TODO: Can we do this without copying?
        # make a new dim for num_actors to do rollouts in parallel
        actors_states = states.reshape([states.shape[0], num_actors, 1])
        # rollout current policy in environment for T timesteps
        cur_log_probs, prev_log_probs, advantages, states = rollout(actors_states)
        ratios = torch.exp(cur_log_probs - prev_log_probs)
        surrogate_loss = policy.compute_surrogate_loss(ratios, advantages, epsilon)
        for k in self.num_epochs:
            optimizer.zero_grad()
            surrogate_loss.backward()
            optimizer.step() 


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
    parser.add_argument('--epsilon', type=int, default=0.2)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    env = rlcard.make(ENV_ID)
    state_channels, action_channels = env_shape(env)
    agent = PPOAgent(state_channels, args.hidden_dim, action_channels)
    init_env(env, agent, args.num_random_agents)

    value_estimator = ValueEstimator()

    train(env, agent, value_estimator, args.num_iters, args.num_actors, args.epsilon, args.lr)