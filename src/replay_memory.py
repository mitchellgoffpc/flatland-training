import random
from collections import namedtuple, deque, Iterable

import torch

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
        self.memory.append(Transition(torch.stack(state, -1).unsqueeze(0),
                                      action,
                                      reward,
                                      torch.stack(next_state, -1).unsqueeze(0),
                                      done))

    def push_episode(self, episode):
        for step in episode.memory:
            self.push(*step)

    def sample(self, batch_size, device):
        experiences = random.sample(self.memory, k=batch_size)

        states = self.stack([e.state for e in experiences]).float().to(device)
        actions = self.stack([e.action for e in experiences]).long().to(device)
        rewards = self.stack([e.reward for e in experiences]).float().to(device)
        next_states = self.stack([e.next_state for e in experiences]).float().to(device)
        dones = self.stack([[v for k, v in e.done.items() if not hasattr(k, 'startswith') or not k.startswith('_')]
                            for e in experiences]).float().to(device)

        return states, actions, rewards, next_states, dones

    def stack(self, states, dim=0):
        if isinstance(states[0], Iterable):
            if isinstance(states[0][0], list):
                return torch.stack([self.stack(st, -1) for st in states], dim)
            if isinstance(states[0], torch.Tensor):
                return torch.stack(states, 0)
            return torch.tensor(states)
        return torch.tensor(states).view(len(states), 1)

    def __len__(self):
        return len(self.memory)
