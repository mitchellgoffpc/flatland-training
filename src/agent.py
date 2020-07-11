import pickle
import random

import torch
from torch_optimizer import Yogi as Optimizer

try:
    from .model import QNetwork, ConvNetwork
    from .replay_memory import ReplayBuffer
except:
    from model import QNetwork, ConvNetwork
    from replay_memory import ReplayBuffer
import os

BUFFER_SIZE = 500_000
BATCH_SIZE = 256
GAMMA = 0.998
TAU = 1e-3
CLIP_FACTOR = 0.2
LR = 2e-4
UPDATE_EVERY = 1
DOUBLE_DQN = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Agent:
    def __init__(self, state_size, action_size, model_depth, hidden_factor, kernel_size, squeeze_heads,
                 use_global=False):
        self.action_size = action_size

        # Q-Network
        if use_global:
            network = ConvNetwork
        else:
            network = QNetwork
        self.policy = network(state_size,
                              action_size,
                              hidden_factor,
                              model_depth,
                              kernel_size,
                              squeeze_heads).to(device)
        self.old_policy = network(state_size,
                                  action_size,
                                  hidden_factor,
                                  model_depth,
                                  kernel_size,
                                  squeeze_heads,
                                  debug=False).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = Optimizer(self.policy.parameters(), lr=LR, weight_decay=1e-2)

        # Replay memory
        self.memory = ReplayBuffer(BATCH_SIZE)
        self.stack = [[] for _ in range(4)]
        self.t_step = 0

    def reset(self):
        self.finished = False

    # Decide on an action to take in the environment

    def act(self, state, eps=0.):
        agent_count = len(state)
        state = torch.stack(state, -1).unsqueeze(0).to(device)
        self.policy.eval()
        with torch.no_grad():
            action_values = self.policy(state)

        # Epsilon-greedy action selection
        return [torch.argmax(action_values[:, :, i], 1).item()
                if random.random() > eps
                else torch.randint(self.action_size, ()).item()
                for i in range(agent_count)]

    def multi_act(self, state, eps=0.):
        agent_count = state.size(-1)
        state = state.to(device)
        self.policy.eval()
        with torch.no_grad():
            action_values = self.policy(state)

        # Epsilon-greedy action selection
        return [[torch.argmax(act[:, i], 0).item()
                 if random.random() > eps
                 else torch.randint(self.action_size, ()).item()
                 for i in range(agent_count)]
                for act in action_values.__iter__()]

    # Record the results of the agent's action and update the model

    def step(self, state, action, agent_done, collision, step_reward=0, collision_reward=-2):
        self.stack[0].append(state)
        self.stack[1].append(action)
        self.stack[2].append([[v for k, v in a.items()
                               if not hasattr(k, 'startswith')
                               or not k.startswith('_')] for a in agent_done])
        self.stack[3].append(collision)

        if len(self.stack) >= UPDATE_EVERY:
            action = torch.tensor(self.stack[1]).flatten(0, 1).to(device)
            reward = torch.tensor([[[1 if ad
                                     else (collision_reward if c else step_reward) for ad, c in zip(ad_batch, c_batch)]
                                    for ad_batch, c_batch in zip(ad_step, c_step)]
                                   for ad_step, c_step in zip(self.stack[2], self.stack[3])]).flatten(0, 1).to(device)
            state = torch.cat(self.stack[0], 0).to(device)
            self.stack = [[] for _ in range(4)]
            self.learn(state, action, reward)

    def learn(self, states, actions, rewards):
        self.policy.train()
        actions.unsqueeze_(1)
        responsible_outputs = self.policy(states).gather(1, actions)
        old_responsible_outputs = self.old_policy(states).gather(1, actions)
        old_responsible_outputs.detach_()
        ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
        clamped_ratio = torch.clamp(ratio, 1. - CLIP_FACTOR, 1. + CLIP_FACTOR)
        loss = -torch.min(ratio * rewards, clamped_ratio * rewards).mean()

        # rewards = rewards - rewards.mean()

        # Compute loss and perform a gradient step
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Checkpointing methods

    def save(self, path, *data):
        torch.save(self.policy.state_dict(), path / 'dqn/model_checkpoint.local')
        torch.save(self.old_policy.state_dict(), path / 'dqn/model_checkpoint.target')
        torch.save(self.optimizer.state_dict(), path / 'dqn/model_checkpoint.optimizer')
        with open(path / 'dqn/model_checkpoint.meta', 'wb') as file:
            pickle.dump(data, file)

    def load(self, path, *defaults):
        loc = {} if torch.cuda.is_available() else {'map_location': torch.device('cpu')}
        try:
            print("Loading model from checkpoint...")
            dqn = os.path.join(path, 'dqn')
            self.policy.load_state_dict(torch.load(os.path.join(dqn, 'model_checkpoint.local'), **loc))
            self.old_policy.load_state_dict(torch.load(os.path.join(dqn, 'model_checkpoint.target'), **loc))
            self.optimizer.load_state_dict(torch.load(os.path.join(dqn, 'model_checkpoint.optimizer'), **loc))
            with open(os.path.join(dqn, 'model_checkpoint.meta'), 'rb') as file:
                return pickle.load(file)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Got exception {exc} loading model data. Possibly no checkpoint found.")
            return defaults
