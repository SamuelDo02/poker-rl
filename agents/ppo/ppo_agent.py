import torch

class PPOAgent:
    def __init__(self): 
        self.use_raw = False
        self.policy = PPOPolicy(self.state_feature_size, self.hidden_dim, self.action_space_size, clip=True)
        self.value_f = ValueEstimator()
        self.gamma = 0.99 # random value

    def step(self, state):
        ''' Predict the action given the current state in generating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent.
        '''
        """action_probs = self.policy(state['obs'])
        action = None
        while action not in list(state['legal_actions'].keys()):
            # keep resampling
            # action is an int / index
            action = torch.multinomial(action_probs, num_samples=1, replacement=True)"""
        action_probs = self.policy(state['obs'])
        mask = torch.zeros_like(action_probs)
        mask[state['legal_actions'].keys()] = 1.0
        masked_probs = action_probs * mask
        total = masked_probs.sum()
        rescaled_action_probs = masked_probs / total
        action = torch.multinomial(rescaled_action_probs, num_samples=1, replacement=True)

        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent.
            probs (list): The list of action probabilities.
        '''
        action_probs = self.policy(state['obs'])
        action = None
        """while action not in list(state['legal_actions'].keys()):
            # keep resampling
            # action is an int / index
            action = torch.multinomial(action_probs, num_samples=1, replacement=True)"""

        mask = torch.zeros_like(action_probs)
        mask[state['legal_actions'].keys()] = 1.0
        masked_probs = action_probs * mask
        total = masked_probs.sum()
        rescaled_action_probs = masked_probs / total
        action = torch.multinomial(rescaled_action_probs, num_samples=1, replacement=True)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: rescaled_action_probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}
        return action, info

    def advantage(self):
        pass
