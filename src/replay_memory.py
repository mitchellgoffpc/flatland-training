import torch
import random
import numpy as np
from collections import namedtuple, deque, Iterable


Transition = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))


class Episode:
    memory = []

    def reset(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(tuple(args))

    def discount_rewards(self, gamma):
        running_add = 0.
        for i, (state, action, reward, *rest) in list(enumerate(self.memory))[::-1]:
            running_add = running_add * gamma + reward
            self.memory[i] = (state, action, running_add, *rest)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done))

    def push_episode(self, episode):
        for step in episode.memory:
            self.push(*step)

    def sample(self, batch_size, device):
        experiences = random.sample(self.memory, k=batch_size)

        states      = torch.from_numpy(self.stack([e.state      for e in experiences])).float().to(device)
        actions     = torch.from_numpy(self.stack([e.action     for e in experiences])).long().to(device)
        rewards     = torch.from_numpy(self.stack([e.reward     for e in experiences])).float().to(device)
        next_states = torch.from_numpy(self.stack([e.next_state for e in experiences])).float().to(device)
        dones       = torch.from_numpy(self.stack([e.done       for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def stack(self, states):
        sub_dims = states[0].shape[1:] if isinstance(states[0], Iterable) else [1]
        return np.reshape(np.array(states), (len(states), *sub_dims))

    def __len__(self):
        return len(self.memory)
