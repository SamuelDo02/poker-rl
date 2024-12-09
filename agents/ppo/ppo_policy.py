import torch
from torch import nn

class PPOPolicy(nn.Module):
    def __init__(self, state_channels, hidden_dim, action_channels, clip=True):
        super(PPOPolicy, self).__init__()
        self.linear1 = nn.Linear(state_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_channels)
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