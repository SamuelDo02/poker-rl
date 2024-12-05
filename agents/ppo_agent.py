import torch

class PPOPolicy:
    # want to make the non-legal probabilities close to zero
    def __init__(self):
        pass
    def forward(self, x):
        pass
    def compute_surrogate_loss(self):
        pass

class ValueEstimator:
    def __init__(self):
        pass
    def forward(self, x):
        pass
    def compute_loss(self):
        pass

class PPOAgent:
    def __init__(self, num_iters, num_timesteps):
        self.use_raw = False
        self.num_iters = num_iters
        self.num_timesteps = num_timesteps
        self.policy = PPOPolicy()
        self.value_f = ValueEstimator()

    def step(state):
        ''' Predict the action given the curent state in generating training data.

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

    def rollout(self, state):
        pass

    def train(self, state):
        for i in range(self.num_actors):
            # rollout policy pi_old in environment for T timesteps
            rewards, advantages, value_f_estimates = self.rollout(state)
        surrogate_loss = self.policy.compute_surrogate_loss(rewards, advantages)
        value_f_loss = self.value_f.compute_loss(rewards, value_f_estimates)
        for k in self.num_epochs:
