from ppo_policy import PPOPolicy

class PPOAgent:
    def __init__(self, state_channels, hidden_dim, action_channels): 
        self.use_raw = False
        self.policy = PPOPolicy(state_channels, hidden_dim, action_channels, clip=True)


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