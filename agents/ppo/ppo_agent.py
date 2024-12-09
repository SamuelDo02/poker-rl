class PPOAgent:
    def __init__(self): 
        self.use_raw = False
        self.policy = PPOPolicy(self.state_feature_size, self.hidden_dim, self.action_space_size, clip=True)
        self.value_f = ValueEstimator()
        self.gamma = 0.99 # random value

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
