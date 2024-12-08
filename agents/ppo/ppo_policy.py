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