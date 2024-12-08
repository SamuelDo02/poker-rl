import rlcard
import torch
import torch.nn as nn

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
        x = self.linear2(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x 

    def compute_surrogate_loss(self, rewards, advantages, epsilon):
        loss = 0
        if self.clip: 
            loss = torch.min(torch.matmul(rewards, advantages), torch.matmul(torch.clip(rewards, 1 - epsilon, 1 + epsilon), advantages))
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
        self.epsilon = 0.05 # for clipping (i just put a random value at this point)
        self.num_actors = num_actors

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
        print(state)
        pass

    def advantage(self):
        pass

    def rollout(self, game_env, policy):
        """
        Rollouts a policy for one round.
        """
        trajectories, payoffs = game_env.run(is_training=True)
        """
        state = game_env.reset()
        returns = []
        rewards = []
        advantages = []
        states = []

        state = rlcard
            while state.game_not_ended:
                action_weights = policy(state)
                action = torch.multinomial(action_weights, 1).item()
                # Below is pseudocode
                state.choose_action(action)

                # This actually has to be computed backwards from the end
                # of the episode to correctly account for discount.
                reward = reward_of(action)
                rewards.append(reward)
                returns.append(return_of(reward, state))
                advantages.append(advantage_of(returns, state))
                states.append(state) # Do we mean value of the state here?
                
        return rewards, advantages, states 
        """

    def train(self, states):
        # make a new dim for num_actors to do rollouts in parallel
        actors_states = states.reshape([states.shape[0], self.num_actors, 1])
        # for i in range(self.num_actors):
            # rollout policy pi_old in environment for T timesteps
            rewards, advantages, states = self.rollout(actors_states, self.gamma, self.num_timesteps)
            surrogate_loss = self.policy.compute_surrogate_loss(rewards, advantages, self.epsilon)
            value_f_loss = self.value_f.compute_loss(states)
        for k in self.num_epochs:
            