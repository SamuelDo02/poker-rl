import torch

from rlcard.games.nolimitholdem.round import Action

from ppo_policy import PPOPolicy

class PPOAgent:
    def __init__(self, state_channels, hidden_dim, action_channels): 
        self.use_raw = False
        self.policy = PPOPolicy(state_channels, hidden_dim, action_channels, clip=True)


    def step_with_probs(self, state, no_grad=False):
        obs = torch.from_numpy(state['obs']).float()

        if no_grad:
            with torch.no_grad():
                action_probs = self.policy(obs)
        else:
            action_probs = self.policy(obs)

        action_idx = torch.multinomial(action_probs, num_samples=1, replacement=True).item()
        action = Action(action_idx)

        # Penalize agent for making non-legal action by always folding.
        if action not in state['raw_legal_actions']:
            action = Action.FOLD 

        return action, action_probs


    def step(self, state):
        ''' Predict the action given the current state in generating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent.
        '''
        action, _ = self.step_with_probs(state)
        return action


    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent.
            probs (list): The list of action probabilities.
        '''
        action, action_probs = self.step_with_probs(state, no_grad=True)

        info = {}
        info['probs'] = { Action(action) : action_probs[action].item() 
                          for action in state['legal_actions'] }

        return action, info