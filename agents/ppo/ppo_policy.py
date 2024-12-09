import torch
from torch import nn

class PPOPolicy(nn.Module):
    def __init__(self, state_channels, hidden_dim, action_channels, clip=True): #, beta=3):
        super(PPOPolicy, self).__init__()
        self.linear1 = nn.Linear(state_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.clip = clip # clipped surrogate loss
        # self.beta = beta # fixed KL penalty surrogate loss, took the best hyperparam value from original paper


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
            loss = torch.min(torch.mul(ratio, advantages), 
                             torch.mul(torch.clip(ratio, 1 - epsilon, 1 + epsilon), 
                                       advantages))
        # else:
        # kl_divergence =
        # loss = torch.mul(ratio, advantages) - self.beta *
        return -loss