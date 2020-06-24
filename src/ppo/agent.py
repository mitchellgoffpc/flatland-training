import pickle
import random
import torch
from torch.distributions.categorical import Categorical

from ppo.model import PolicyNetwork
from replay_memory import Episode, ReplayBuffer

BUFFER_SIZE = 32_000
BATCH_SIZE = 512
GAMMA = 0.9
LR = 0.5e-4
CLIP_FACTOR = .1
UPDATE_EVERY = 40

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, num_agents):
        self.policy = PolicyNetwork(state_size, action_size).to(device)
        self.old_policy = PolicyNetwork(state_size, action_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)

        self.episodes = [Episode() for _ in range(num_agents)]
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.t_step = 0


    # Decide on an action to take in the environment
    def act(self, state, eps=None):
        self.policy.eval()
        with torch.no_grad():
            output = self.policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
            return Categorical(output).sample().item()


    # Record the results of the agent's action and update the model
    def step(self, handle, state, action, reward, next_state, done, train=True):
        reward = 1 if reward > 0 else 0
        self.episodes[handle].push(state, action, reward, next_state, done)

        if done:
            self.episodes[handle].discount_rewards(GAMMA)
            self.memory.push_episode(self.episodes[handle])
            self.episodes[handle].reset()

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if train and len(self.memory) > BATCH_SIZE * 4 and self.t_step == 0:
            self.learn(*self.memory.sample(BATCH_SIZE, device))

    def learn(self, states, actions, rewards, next_state, done):
        self.policy.train()

        responsible_outputs = torch.gather(self.policy(states), 1, actions)
        old_responsible_outputs = torch.gather(self.old_policy(states), 1, actions).detach()

        ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
        clamped_ratio = torch.clamp(ratio, 1. - CLIP_FACTOR, 1. + CLIP_FACTOR)
        loss = -torch.min(ratio * rewards, clamped_ratio * rewards).mean()

        # Compute loss and perform a gradient step
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # Checkpointing functions

    def save(self, path, *data):
        torch.save(self.policy.state_dict(), path / 'ppo/model_checkpoint.policy')
        torch.save(self.optimizer.state_dict(), path / 'ppo/model_checkpoint.optimizer')
        with open(path / 'ppo/model_checkpoint.meta', 'wb') as file:
            pickle.dump(data, file)

    def load(self, path, *defaults):
        try:
            print("Loading model from checkpoint...")
            self.policy.load_state_dict(torch.load(path / 'ppo/model_checkpoint.policy'))
            self.optimizer.load_state_dict(torch.load(path / 'ppo/model_checkpoint.optimizer'))
            with open(path / 'ppo/model_checkpoint.meta', 'rb') as file:
                return pickle.load(file)
        except:
            print("No checkpoint file was found")
            return defaults
