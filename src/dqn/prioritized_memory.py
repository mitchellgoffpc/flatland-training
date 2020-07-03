# Copied from https://github.com/rlcode/per/blob/master/prioritized_memory.py

import torch
import numpy as np
from dqn.sum_tree import SumTree


# Hyperparameters
epsilon = 0.01    # Small constant added to each priority to ensure all of them are sampled with p > 0
alpha = 0.6       # Exponent for interpolating between greedy selection and uniform selection
beta_start = 0.4  # Exponent for importance sampling
beta_increment_per_sampling = 0.00001 # Decay schedule for beta

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.beta = beta_start

    def get_priority(self, error):
        return (abs(error) + epsilon) ** alpha

    def push(self, error, *sample):
        # self.memory.append(sample)
        p = self.get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size, device):
        samples, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size

        self.beta = min(1, self.beta + beta_increment_per_sampling)

        # indices = np.random.choice(len(self), batch_size)
        # samples = [self.memory[i] for i in indices]
        # priorities = np.ones(len(self))
        # is_weight = np.ones(len(self)) / len(self)

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            index, priority, data = self.tree.get(np.random.uniform(a, b))
            samples.append(data)
            indices.append(index)
            priorities.append(priority)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return tuple(torch.from_numpy(np.stack(x)).to(device) for x in zip(*samples)) + (indices, torch.from_numpy(is_weight))

    def update(self, idx, error):
        p = self.get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        # return len(self.memory)
        return self.tree.n_entries
