import copy
import pickle
import random

import torch
import torch.nn.functional as F
from torch_optimizer import Yogi as Optimizer

try:
    from .model import QNetwork
    from .replay_memory import ReplayBuffer
except:
    from model import QNetwork
    from replay_memory import ReplayBuffer
import os

BUFFER_SIZE = 500_000
BATCH_SIZE = 512
GAMMA = 0.998
TAU = 1e-3
LR = 3e-5
UPDATE_EVERY = 160

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, num_agents, model_depth, hidden_factor, kernel_size, squeeze_heads):
        self.action_size = action_size

        # Q-Network
        self.qnetwork_local = QNetwork(state_size,
                                       action_size,
                                       hidden_factor,
                                       model_depth,
                                       kernel_size,
                                       squeeze_heads).to(device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.optimizer = Optimizer(self.qnetwork_local.parameters(), lr=LR, weight_decay=1e-2)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.t_step = 0

    def reset(self):
        self.finished = False

    # Decide on an action to take in the environment

    def act(self, state, eps=0.):
        agent_count = len(state)
        state = torch.stack(state, -1).unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        return [torch.argmax(action_values[:, :, i], 1).item()
                if random.random() > eps
                else torch.randint(self.action_size, ()).item()
                for i in range(agent_count)]

    # Record the results of the agent's action and update the model

    def step(self, state, action, next_state, agent_done, episode_done, collision, step_reward=-1):
        if not self.finished:
            if agent_done:
                reward = 1
            elif collision:
                reward = -5
            else:
                reward = step_reward

            # Save experience in replay memory
            self.memory.push(state, action, reward, next_state, agent_done or episode_done)
            self.finished = episode_done

        # Perform a gradient update every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE * 20:
            self.learn(*self.memory.sample(BATCH_SIZE, device))

    def learn(self, states, actions, rewards, next_states, dones):
        self.qnetwork_local.train()


        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states.squeeze(1))

        Q_expected = Q_expected.gather(1, actions.unsqueeze(1))
        Q_best_action = self.qnetwork_local(next_states.squeeze(1)).argmax(1)
        Q_targets_next = self.qnetwork_target(next_states.squeeze(1)).gather(1, Q_best_action.unsqueeze(1))

        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(-1) + GAMMA * Q_targets_next * (1 - dones.unsqueeze(-1))

        # Compute loss and perform a gradient step
        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_expected, Q_targets)
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
