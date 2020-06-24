import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidsize1)
        self.fc2 = nn.Linear(hidsize1, hidsize2)
        self.output = nn.Linear(hidsize2, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.float()))
        x = F.relu(self.fc2(x))
        return self.softmax(self.output(x))
