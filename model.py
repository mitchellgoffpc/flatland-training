import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidsize1=256, hidsize2=128):
        super(QNetwork, self).__init__()

        # self.conv1_val = nn.Conv2d(6, 10, 5, stride=2, padding=1, bias=False)
        # self.conv2_val = nn.Conv2d(10, 20, 3, bias=False)
        # self.fc1_val = nn.Linear(80, 64)
        # self.fc2_val = nn.Linear(64, 1)
        # self.bn1_val = nn.BatchNorm2d(10)
        # self.bn2_val = nn.BatchNorm2d(20)
        #
        # self.conv1_adv = nn.Conv2d(6, 10, 5, stride=2, padding=1, bias=False)
        # self.conv2_adv = nn.Conv2d(10, 20, 3, bias=False)
        # self.fc1_adv = nn.Linear(80, 64)
        # self.fc2_adv = nn.Linear(64, action_size)
        # self.bn1_adv = nn.BatchNorm2d(10)
        # self.bn2_adv = nn.BatchNorm2d(20)

        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc3_val = nn.Linear(hidsize2, 1)

        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc3_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc3_adv(adv)
        return val + adv - adv.mean()

        # x = x.permute(0, 3, 1, 2)
        #
        # val = F.relu(self.bn1_val(self.conv1_val(x)))
        # val = F.relu(self.bn2_val(self.conv2_val(val)))
        # val = F.relu(self.fc1_val(val.view(val.shape[0], -1)))
        # val = self.fc2_val(val)
        #
        # adv = F.relu(self.bn1_adv(self.conv1_adv(x)))
        # adv = F.relu(self.bn2_adv(self.conv2_adv(adv)))
        # adv = F.relu(self.fc1_adv(adv.view(adv.shape[0], -1)))
        # adv = self.fc2_adv(adv)
        #
        # return val + adv - adv.mean()
