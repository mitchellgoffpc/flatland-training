import math
import pickle

import numpy as np
import torch
from torch_optimizer import Yogi as Optimizer

try:
    from .model import QNetwork, ConvNetwork, init, GlobalStateNetwork
except:
    from model import QNetwork, ConvNetwork, init, GlobalStateNetwork
import os

BATCH_SIZE = 256
CLIP_FACTOR = 0.2
LR = 1e-4
UPDATE_EVERY = 1
CUDA = True
MINI_BACKWARD = False
DQN = True
DQN_PARAMS = {'tau': 1e-3, 'gamma': 0.998}

device = torch.device("cuda:0" if CUDA and torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, model_depth, hidden_factor, kernel_size, squeeze_heads, decoder_depth,
                 model_type=0, softmax=True, debug=True):
        self.action_size = action_size

        # Q-Network
        if model_type == 1:  # Global/Local
            network = ConvNetwork
        elif model_type == 0:  # Tree
            network = QNetwork
        else:  # Global State
            network = GlobalStateNetwork
        self.policy = network(state_size,
                              action_size,
                              hidden_factor,
                              model_depth,
                              kernel_size,
                              squeeze_heads,
                              decoder_depth,
                              softmax=softmax).to(device)
        self.old_policy = network(state_size,
                                  action_size,
                                  hidden_factor,
                                  model_depth,
                                  kernel_size,
                                  squeeze_heads,
                                  decoder_depth,
                                  softmax=softmax,
                                  debug=False).to(device)
        if debug:
            print(self.policy)

            parameters = sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.policy.parameters()))
            digits = int(math.log10(parameters))
            number_string = " kMGTPEZY"[digits // 3]

            print(f"[DEBUG/MODEL] Training with {parameters * 10 ** -(digits // 3 * 3):.1f}"
                  f"{number_string} parameters")
        self.policy.apply(init)
        try:
            self.policy = torch.jit.script(self.policy)
            self.old_policy = torch.jit.script(self.old_policy)
        except:
            import traceback
            traceback.print_exc()
            print("NO JIT")
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = Optimizer(self.policy.parameters(), lr=LR, weight_decay=1e-2)

        # Replay memory
        self.stack = [[] for _ in range(6)]
        self.t_step = 0
        self.idx = 1

    def reset(self):
        self.policy.reset_cache()
        self.old_policy.reset_cache()

    def multi_act(self, state):
        self.policy.eval()
        with torch.no_grad():
            action_values = self.policy(*state)
        return action_values.argmax(1).detach().cpu().numpy()

    def step(self, state, action, agent_done, collision, next_state, step_reward=0, collision_reward=-2):
        agent_count = len(agent_done[0])-1
        self.stack[0].append(state)
        self.stack[1].append(action)
        self.stack[2].append([[done[idx] for idx in range(agent_count)] for done in agent_done])
        self.stack[3].append(collision)
        self.stack[4].append(next_state)

        if MINI_BACKWARD or len(self.stack[0]) >= UPDATE_EVERY:
            action = torch.tensor(self.stack[1]).flatten(0, 1).to(device)
            agent_done = np.array(self.stack[2])
            collision = np.array(self.stack[3])
            reward = np.where(agent_done, 1, np.where(collision, collision_reward, step_reward))
            reward = torch.tensor(reward, device=device, dtype=torch.float).flatten(0, 1)
            state = tuple(torch.cat(st, 0) for st in zip(*self.stack[0]))
            next_state = tuple(torch.cat(st, 0) for st in zip(*self.stack[4]))
            agent_done = torch.as_tensor(agent_done, device=device, dtype=torch.int8)
            self.stack = [[] for _ in range(6)]
            self.learn(state, action, reward, next_state, agent_done)

    def learn(self, states, actions, rewards, next_states, done):
        if MINI_BACKWARD:
            self.idx = (self.idx + 1) % UPDATE_EVERY

        self.policy.train()
        actions.unsqueeze_(1)

        if DQN:
            expected = self.policy(*states).gather(1, actions)
            best_action = self.policy(*next_states).argmax(1)
            targets_next = self.old_policy(*next_states).gather(1, best_action.unsqueeze(1))
            targets = rewards + DQN_PARAMS['gamma'] * targets_next * (1 - done)
            loss = (expected - targets).square().max(0)[0].mean()
        else:
            with torch.no_grad():
                states_clone = tuple(st.clone().detach().requires_grad_(False) for st in states)
                old_responsible_outputs = self.old_policy(*states_clone).gather(1, actions)
            old_responsible_outputs.detach_()
            responsible_outputs = self.policy(*states).gather(1, actions)
            ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
            ratio.squeeze_(1)
            clamped_ratio = torch.clamp(ratio, 1. - CLIP_FACTOR, 1. + CLIP_FACTOR)
            loss = torch.min(ratio * rewards, clamped_ratio * rewards).sum(-1).max(0)[0].mean().neg()

        if MINI_BACKWARD:
            loss = loss / UPDATE_EVERY

        loss.backward()

        if not MINI_BACKWARD or self.idx == 0:
            if not DQN:
                self.old_policy.load_state_dict(self.policy.state_dict())
            self.optimizer.step()
            self.optimizer.zero_grad()
            if DQN:
                for target_param, local_param in zip(self.old_policy.parameters(), self.policy.parameters()):
                    target_param.data.copy_(DQN_PARAMS['tau'] * local_param.data +
                                            (1.0 - DQN_PARAMS['tau']) * target_param.data)


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
