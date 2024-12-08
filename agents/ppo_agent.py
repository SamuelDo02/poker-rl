import torch
import torch.nn as nn
from tqdm import tqdm

class PPOPolicy:
    # want to make the non-legal probabilities close to zero
    def __init__(self, in_channels, hidden_dim, action_space, clip=True):
        # in channels = state feature size 
        self.linear1 = nn.Linear(in_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_space)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.clip = clip 

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x 

    def compute_surrogate_loss(self, ratio, advantages, epsilon):
        loss = 0
        if self.clip:
            loss = torch.min(torch.matmul(ratio, advantages), torch.matmul(torch.clip(ratio, 1 - epsilon, 1 + epsilon), advantages))
        return loss 

class ValueEstimator:
    def __init__(self):
        self.linear1 = nn.Linear(in_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

    def compute_loss(self, states):

        pass

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
        pass

    def advantage(self):
        pass

    def rollout(self, state, num_games):
        # TODO: None of this works. This is just pseudo code.
        # sample the next state for T timesteps 
        returns = []
        rewards = []
        advantages = []
        states = []

        for _ in range(num_games):
            while state.game_not_ended:
                action_distr = self.policy(state)
                action = torch.sample(action_distr)
                state.choose_action(action)

                # This actually has to be computed backwards from the end
                # of the episode to correctly account for discount.
                reward = reward_of(action)
                rewards.append(reward)
                returns.append(return_of(reward, state))
                advantages.append(advantage_of(returns, state))
                states.append(state) # Do we mean value of the state here?
                
        return rewards, advantages, states 

    def train(self, states):
        for i in tqdm(range(self.num_iters)):
            # make a new dim for num_actors to do rollouts in parallel
            actors_states = states.reshape([states.shape[0], self.num_actors, 1])
            # rollout current policy in environment for T timesteps
            cur_log_probs, prev_log_probs, advantages, states = self.rollout(actors_states, self.num_timesteps)
            ratios = torch.exp(cur_log_probs - prev_log_probs)
            surrogate_loss = self.policy.compute_surrogate_loss(ratios, advantages, self.epsilon)
            for k in self.num_epochs:
                surrogate_loss.backward()
            self.old_policy = self.policy
