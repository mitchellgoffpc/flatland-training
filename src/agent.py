import math
import pickle
import typing

import numpy as np
import torch
from torch_optimizer import Yogi as Optimizer

try:
    from .model import QNetwork, ConvNetwork, init, GlobalStateNetwork, TripleClassificationHead, NAFHead
except:
    from model import QNetwork, ConvNetwork, init, GlobalStateNetwork, TripleClassificationHead, NAFHead
import os

BATCH_SIZE = 256
CLIP_FACTOR = 0.2
LR = 1e-4
UPDATE_EVERY = 1
CUDA = True
MINI_BACKWARD = False
DQN_TAU = 1e-3
EPOCHS = 1

device = torch.device("cuda:0" if CUDA and torch.cuda.is_available() else "cpu")


@torch.jit.script
def aggregate(loss: torch.Tensor):
    maximum, _ = loss.max(0)
    return maximum.sum()


@torch.jit.script
def mse(in_x, in_y):
    return aggregate((in_x - in_y).square())


@torch.jit.script
def dqn_target(rewards, targets_next, done):
    return rewards + 0.998 * targets_next * (1 - done)


class Agent(torch.nn.Module):
    def __init__(self, state_size, action_size, model_depth, hidden_factor, kernel_size, squeeze_heads, decoder_depth,
                 model_type=0, softmax=True, debug=True, loss_type='PPO'):
        super(Agent, self).__init__()
        self.action_size = action_size

        # Q-Network
        if model_type == 1:  # Global/Local
            network = ConvNetwork
        elif model_type == 0:  # Tree
            network = QNetwork
        else:  # Global State
            network = GlobalStateNetwork
        if loss_type in ('PPO', 'DQN'):
            tail = TripleClassificationHead(hidden_factor, action_size)
        else:
            tail = NAFHead(hidden_factor, action_size)
        self.policy = network(state_size,
                              hidden_factor,
                              model_depth,
                              kernel_size,
                              squeeze_heads,
                              decoder_depth,
                              tail=tail,
                              softmax=softmax).to(device)
        self.old_policy = network(state_size,
                                  hidden_factor,
                                  model_depth,
                                  kernel_size,
                                  squeeze_heads,
                                  decoder_depth,
                                  tail=tail,
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
        self.tensor_stack = []
        self._policy_update = loss_type in ("PPO",)
        self._soft_update = loss_type in ("DQN", "NAF")

        self._action_index = torch.zeros(1)
        self._value_index = torch.zeros(1) + 1
        self._triangular_index = torch.zeros(1) + 2

        self.loss = getattr(self, f'_{loss_type.lower()}_loss')

    def _dqn_loss(self, states, actions, next_states, rewards, done):
        actions = actions.argmax(1)
        expected = self.policy(self._action_index, self._action_index, *states).gather(1, actions)
        best_action = self.policy(self._action_index, self._action_index, next_states).argmax(1)
        targets_next = self.old_policy(self._action_index, self._action_index,
                                       *next_states).gather(1, best_action.unsqueeze(1))
        targets = dqn_target(rewards, targets_next, done)
        loss = mse(expected, targets)
        return loss

    def _naf_loss(self, states, actions, next_states, rewards, done):
        targets_next = self.old_policy(self._value_index, self._action_index, next_states)
        state_action_values = self.policy(self._triangular_index, actions, states)
        targets = dqn_target(rewards, targets_next, done)
        loss = mse(state_action_values, targets)
        return loss

    def _ppo_loss(self, states, actions, next_states, rewards, done):
        _ = next_states
        _ = done
        actions = actions.argmax(1)
        states_clone = [st.clone().detach().requires_grad_(False) for st in states]
        old_responsible_outputs = self.old_policy(self._action_index, self._action_index,
                                                  *states_clone).gather(1, actions).detach_()
        responsible_outputs = self.policy(self._action_index, self._action_index, *states).gather(1, actions)
        ratio = responsible_outputs / (old_responsible_outputs + 1e-5)
        clamped_ratio = torch.clamp(ratio, 1. - CLIP_FACTOR, 1. + CLIP_FACTOR)
        loss = aggregate(torch.min(ratio * rewards, clamped_ratio * rewards)).neg()
        return loss

    def reset(self):
        self.policy.reset_cache()
        self.old_policy.reset_cache()

    def multi_act(self, state, argmax_only=True) -> typing.Union[typing.Tuple[np.ndarray, np.ndarray], np.ndarray]:
        self.policy.eval()
        with torch.no_grad():
            action_values = self.policy(self._action_index, self._action_index, *state).detach()
        argmax = action_values.argmax(1).cpu().numpy()
        if argmax_only:
            return argmax
        return action_values, argmax

    def step(self, state, action, agent_done, collision, next_state, step_reward=0, collision_reward=-2):
        agent_count = len(agent_done[0]) - 1
        self.stack[0].append(state)
        self.stack[1].append(action)
        self.stack[2].append([[done[idx] for idx in range(agent_count)] for done in agent_done])
        self.stack[3].append(collision)
        self.stack[4].append(next_state)

        if MINI_BACKWARD or len(self.stack[0]) >= UPDATE_EVERY:
            action = torch.cat(self.stack[1]).to(device).unsqueeze_(1)
            agent_done = np.array(self.stack[2])
            collision = np.array(self.stack[3])
            reward = np.where(agent_done, 1, np.where(collision, collision_reward, step_reward))
            reward = torch.tensor(reward, device=device, dtype=torch.float).flatten(0, 1).unsqueeze_(1)
            state = tuple(torch.cat(st, 0) for st in zip(*self.stack[0]))
            next_state = tuple(torch.cat(st, 0) for st in zip(*self.stack[4]))
            agent_done = torch.as_tensor(agent_done, device=device, dtype=torch.int8).flatten(0, 1).unsqueeze_(1)
            self.stack = [[] for _ in range(6)]
            self.tensor_stack.append((state, action, reward, next_state, agent_done))
            if len(self.tensor_stack) >= EPOCHS:
                tensor_stack = (torch.cat(t, 0) if isinstance(t[0], torch.Tensor)
                                else tuple(torch.cat(sub_t, 0) for sub_t in zip(*t))
                                for t in zip(*self.tensor_stack))
                del self.tensor_stack[0]
                self.learn(*tensor_stack)

    def learn(self, states, actions, rewards, next_states, done):
        if MINI_BACKWARD:
            self.idx = (self.idx + 1) % UPDATE_EVERY

        self.policy.train()

        loss = self.loss(states, actions, next_states, rewards, done)

        if MINI_BACKWARD:
            loss = loss / UPDATE_EVERY

        loss.backward()

        if not MINI_BACKWARD or self.idx == 0:
            if self._policy_update:
                self.old_policy.load_state_dict(self.policy.state_dict())
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self._soft_update:
                for target_param, local_param in zip(self.old_policy.parameters(), self.policy.parameters()):
                    target_param.data.copy_(DQN_TAU * local_param.data +
                                            (1.0 - DQN_TAU) * target_param.data)

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
