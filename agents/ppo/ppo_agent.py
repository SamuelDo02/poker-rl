import rlcard
import torch
import torch.nn as nn
from tqdm import tqdm

class PPOAgent:
    def __init__(self, num_iters, num_timesteps, num_epochs, epsilon, num_actors):
        self.use_raw = False
        self.num_iters = num_iters
        self.num_timesteps = num_timesteps
        self.num_epochs = num_epochs 
        self.policy = PPOPolicy(self.state_feature_size, self.hidden_dim, self.action_space_size, clip=True)
        self.value_f = ValueEstimator()
        self.epsilon = 0.2 # for clipping
        self.num_actors = num_actors
        self.gamma = 0.99 # random value
        self.old_policy = # store the action distribution from 1 step prior

    def step(state):
        ''' Predict the action given the current state in generating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent.
        '''
        pass

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted predicted by the agent.
            probs (list): The list of action probabilities.
        '''
        print(state)
        pass

    def advantage(self):
        pass
