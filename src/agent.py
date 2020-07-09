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
BATCH_SIZE = 64
GAMMA = 0.998
TAU = 1e-3
LR = 2e-4
UPDATE_EVERY = 1

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
        self.qnetwork_local = network(state_size,
                                      action_size,
                                      hidden_factor,
                                      model_depth,
                                      kernel_size,
                                      squeeze_heads).to(device)
        self.qnetwork_target = network(state_size,
                                       action_size,
                                       hidden_factor,
                                       model_depth,
                                       kernel_size,
                                       squeeze_heads,
                                       debug=False).to(device)
        self.optimizer = Optimizer(self.qnetwork_local.parameters(), lr=LR, weight_decay=1e-2)

        # Replay memory
        self.memory = ReplayBuffer(BATCH_SIZE)
        self.t_step = 0

    def reset(self):
        self.finished = False

    # Decide on an action to take in the environment

    def act(self, state, eps=0.):
        agent_count = len(state)
        state = torch.stack(state, -1).unsqueeze(0).to(device)
        state = torch.cat([state, torch.randn(1, 1, state.size(-1), device=device)], 1)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        return [torch.argmax(action_values[:, :, i], 1).item()
                if random.random() > eps
                else torch.randint(self.action_size, ()).item()
                for i in range(agent_count)]

    def multi_act(self, states, eps=0.):
        agent_count = len(states[0])
        state = torch.stack([torch.stack(state, -1) if len(state) > 1 else state.unsqueeze(-1)
                             for state in states], 0).to(device)
        state = torch.cat([state, torch.randn(state.size(0), 1, state.size(-1), device=device)], 1)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        return [[torch.argmax(act[:, i], 0).item()
                 if random.random() > eps
                 else torch.randint(self.action_size, ()).item()
                 for i in range(agent_count)]
                for act in action_values.__iter__()]

    # Record the results of the agent's action and update the model

    def step(self, state, action, next_state, agent_done, episode_done, collision, step_reward=-1):
        state = self.memory.stack(state).to(device).transpose(1, 2)
        action = self.memory.stack(action).to(device)
        reward = self.memory.stack([1 if ad
                                    else (c - 5 if collision else step_reward)
                                    for ad, c in zip(agent_done, collision)]).to(device)
        next_state = self.memory.stack(next_state).to(device).transpose(1, 2)
        dones = self.memory.stack([[v or episode_done for k, v in a.items()
                                    if not hasattr(k, 'startswith')
                                    or not k.startswith('_')] for a in agent_done]).to(device).float()
        state = torch.cat([state, torch.randn(state.size(0), 1, state.size(-1), device=device)], 1)
        next_state = torch.cat([next_state, torch.randn(state.size(0), 1, state.size(-1), device=device)], 1)
        self.learn(state, action, reward, next_state, dones)

    def learn(self, states, actions, rewards, next_states, dones):
        self.qnetwork_local.train()

        actions.squeeze_(-1)
        dones.squeeze_(-1)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states.squeeze(1))

        Q_expected = Q_expected.gather(1, actions.unsqueeze(1))
        Q_best_action = self.qnetwork_local(next_states.squeeze(1)).argmax(1)
        Q_targets_next = self.qnetwork_target(next_states.squeeze(1)).gather(1, Q_best_action.unsqueeze(1))

        # Compute loss and perform a gradient step
        self.optimizer.zero_grad()
        loss = (rewards.unsqueeze(-1) + GAMMA * Q_targets_next * (1 - dones.unsqueeze(-2)) - Q_expected).square().mean()
        loss.backward()
        self.optimizer.step()

        # Update the target network parameters to `tau * local.parameters() + (1 - tau) * target.parameters()`
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    # Checkpointing methods

    def save(self, path, *data):
        torch.save(self.qnetwork_local.state_dict(), path / 'dqn/model_checkpoint.local')
        torch.save(self.qnetwork_target.state_dict(), path / 'dqn/model_checkpoint.target')
        torch.save(self.optimizer.state_dict(), path / 'dqn/model_checkpoint.optimizer')
        with open(path / 'dqn/model_checkpoint.meta', 'wb') as file:
            pickle.dump(data, file)

    def load(self, path, *defaults):
        loc = {} if torch.cuda.is_available() else {'map_location': torch.device('cpu')}
        try:
            print("Loading model from checkpoint...")
            dqn = os.path.join(path, 'dqn')
            self.qnetwork_local.load_state_dict(torch.load(os.path.join(dqn, 'model_checkpoint.local'), **loc))
            self.qnetwork_target.load_state_dict(torch.load(os.path.join(dqn, 'model_checkpoint.target'), **loc))
            self.optimizer.load_state_dict(torch.load(os.path.join(dqn, 'model_checkpoint.optimizer'), **loc))
            with open(os.path.join(dqn, 'model_checkpoint.meta'), 'rb') as file:
                return pickle.load(file)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"Got exception {exc} loading model data. Possibly no checkpoint found.")
            return defaults
