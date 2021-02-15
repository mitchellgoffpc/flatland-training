import copy
import random
import pickle
import torch
import torch.nn.functional as F

from dqn.model import QNetwork
from replay_memory import ReplayBuffer

BUFFER_SIZE = 500_000
BATCH_SIZE = 512
GAMMA = 0.998
TAU = 1e-3
LR = 0.5e-4
UPDATE_EVERY = 40

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, num_agents, double_dqn=True):
        self.action_size = action_size
        self.double_dqn = double_dqn

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.num_agents = num_agents
        self.t_step = 0

    def reset(self):
        self.finished = [False] * self.num_agents


    # Decide on an action to take in the environment

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
              return torch.argmax(action_values).item()
        else: return torch.randint(self.action_size, ()).item()


    # Record the results of the agent's action and update the model

    def step(self, handle, state, action, next_state, agent_done, episode_done, collision):
        if not self.finished[handle]:
            if agent_done:
                  reward = 1
            elif collision:
                  reward = -5
            else: reward = -.1

            # Save experience in replay memory
            self.memory.push(state, action, reward, next_state, agent_done or episode_done)
            self.finished[handle] = agent_done or episode_done

        # Perform a gradient update every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE * 20:
            self.train(*self.memory.sample(BATCH_SIZE, device))


    def train(self, states, actions, rewards, next_states, dones):
        self.qnetwork_local.train()

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
              Q_best_action = self.qnetwork_local(next_states).argmax(1)
              Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_best_action.unsqueeze(-1))
        else: Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        Q_targets = rewards + GAMMA * Q_targets_next * (1 - dones)

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
        try:
            print("Loading model from checkpoint...")
            self.qnetwork_local.load_state_dict(torch.load(path / 'dqn/model_checkpoint.local'))
            self.qnetwork_target.load_state_dict(torch.load(path / 'dqn/model_checkpoint.target'))
            self.optimizer.load_state_dict(torch.load(path / 'dqn/model_checkpoint.optimizer'))
            with open(path / 'dqn/model_checkpoint.meta', 'rb') as file:
                return pickle.load(file)
        except:
            print("No checkpoint file was found")
            return defaults
