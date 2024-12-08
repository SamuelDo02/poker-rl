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