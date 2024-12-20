''' 
An example of evluating the trained models in RLCard
Code adapted from RLCard repo.
'''
import os
import argparse
from ppo_agent import PPOAgent
from loader import load_model

import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
)
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)
import matplotlib.pyplot as plt
import re

def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)


def evaluate_model_checkpoints(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    average_rewards = {}
    iterations = []
    ckpt_path = args.ckpts
    print(os.listdir(ckpt_path))
    for ckpt_file in sorted(os.listdir(ckpt_path)):
        model_path = os.path.join(ckpt_path, ckpt_file)
        print(model_path)
        model_paths = [model_path, args.opp_path]
        if model_path.endswith(".pt"):
            match = re.search(r'\d+', os.path.basename(model_path))
            iteration = int(match.group())
            iterations.append(iteration)
            # Load models
            agents = []
            for position, model in enumerate(model_paths):
                agents.append(load_model(model, env, position, device))
            env.set_agents(agents)

            # Evaluate
            rewards = tournament(env, args.num_games)
            for position, reward in enumerate(rewards):
                print(position, model_paths[position], reward)
                if position == 0:
                    average_rewards[iteration] = reward

    ave_reward_path = f'{ckpt_path}/average_rewards_{args.opp}.png'
    sorted_rewards = []
    for iteration in sorted(iterations):
        sorted_rewards.append(average_rewards[iteration])
    plt.plot(sorted(iterations), sorted_rewards)
    plt.xlabel('Iterations')
    plt.ylabel(f'Average Reward over {args.num_games} games')
    plt.savefig(ave_reward_path)
    plt.close()

def tournament_win_rate(env, num):
    ''' Evaluate he win rate of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of average payoffs for each player
    '''
    payoffs = [[] for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i].append(_payoffs[i])
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i].append(_payoffs[i])
            counter += 1
    win_rates = [0 for _ in range(env.num_players)]
    for i, _ in enumerate(payoffs):
        positive_payoffs = [1 if x > 0 else 0 for x in payoffs[i]]
        win_rates[i] = sum(positive_payoffs) / counter

    return win_rates

def evaluate_win_rate(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    win_rates = tournament_win_rate(env, args.num_games)

    for position, win_rate in enumerate(win_rates):
        print(position, args.models[position], win_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            'experiments/leduc_holdem_dqn_result/model.pth',
            'random',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=10000,
    )

    parser.add_argument(
        '--ckpts',
        type=str,
        default="models/ppo/default_num_iters_10000_correct/checkpoints",
    )
    parser.add_argument(
        '--all',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--opp_path',
        type=str,
        default="random",
    )
    parser.add_argument(
        '--opp',
        type=str,
        default="random",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.all:
        evaluate_model_checkpoints(args)
    else:
        evaluate(args)



