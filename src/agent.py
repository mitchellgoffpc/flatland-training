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

BUFFER_SIZE = 500_000
BATCH_SIZE = 256
GAMMA = 0.998
TAU = 1e-3
CLIP_FACTOR = 0.2
LR = 1e-4
UPDATE_EVERY = 1
DOUBLE_DQN = False
CUDA = True

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
        self.stack = [[] for _ in range(4)]
        self.t_step = 0

    def reset(self):
        self.policy.reset_cache()
        self.old_policy.reset_cache()

    def multi_act(self, state):
        if isinstance(state, tuple):
            state = tuple(s.to(device) for s in state)
        elif isinstance(state, torch.Tensor):
            state = (state.to(device),)
        self.policy.eval()
        with torch.no_grad():
            action_values = self.policy(*state)

        # Epsilon-greedy action selection
        return action_values.argmax(1).detach().cpu().numpy()

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
            state = self.stack[0]
            if isinstance(state[0], tuple):
                state = zip(*state)
                state = tuple(torch.cat(st, 0).to(device) for st in state)
            elif isinstance(state[0], torch.Tensor):
                state = (torch.cat(state, 0).to(device),)
            self.stack = [[] for _ in range(4)]
            self.learn(state, action, reward)

    def learn(self, states, actions, rewards):
        self.policy.train()
        actions.unsqueeze_(1)

        with torch.no_grad():
            states_clone = tuple(st.clone() for st in states)
            for st in states:
                st.requires_grad_(False)
            old_responsible_outputs = self.old_policy(*states_clone).gather(1, actions)
        old_responsible_outputs.detach_()
        responsible_outputs = self.policy(*states).gather(1, actions)
        ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
        ratio.squeeze_(1)
        clamped_ratio = torch.clamp(ratio, 1. - CLIP_FACTOR, 1. + CLIP_FACTOR)
        loss = -torch.min(ratio * rewards, clamped_ratio * rewards).sum(-1).mean()
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
