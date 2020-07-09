import numpy as np
import torch

try:
    from .tree_observation import ACTIONS
except:
    from tree_observation import ACTIONS

ZERO_NODE = torch.zeros((11,))


# Helper function to detect collisions
def is_collision(obs):
    return obs is not None \
           and isinstance(obs.childs['L'], float) \
           and isinstance(obs.childs['R'], float) \
           and obs.childs['F'].num_agents_opposite_direction > 0 \
           and obs.childs['F'].dist_other_agent_encountered <= 1 \
           and obs.childs['F'].dist_other_agent_encountered < obs.childs['F'].dist_unusable_switch
    # and obs.childs['F'].dist_other_agent_encountered < obs.childs['F'].dist_to_next_branch


# Recursively create numpy arrays for each tree node
def create_tree_features(node, current_depth, max_depth, empty_node, data):
    if node == -np.inf or node is None:
        num_remaining_nodes = (4 ** (max_depth - current_depth + 1) - 1) // (4 - 1)
        data.extend([empty_node] * num_remaining_nodes)

    else:
        data.append(torch.FloatTensor(tuple(node)[:-2]))
        if node.childs:
            for direction in ACTIONS:
                create_tree_features(node.childs[direction], current_depth + 1, max_depth, empty_node, data)

    return data


TRUE = torch.ones(1)
FALSE = torch.zeros(1)


# Normalize an observation to [0, 1] and then clip it to get rid of any infinite-valued features
@torch.jit.script
def max_obs(obs):
    out = obs[obs < 1000].max()
    out.clamp_(min=1)
    out.add_(1)
    return out


@torch.jit.script
def wrap(data: torch.Tensor):
    start = data[:, :6]
    mid = data[:, 6]
    max0 = max_obs(start)
    max1 = max_obs(mid)

    min_obs = mid[mid >= 0].min()

    mid.sub_(min_obs)
    max1.sub_(min_obs)
    mid.div_(max1)

    start.div_(max0)

    data.clamp_(-1, 1)

    data[:, :6].sub_(data[:, :6].mean())
    data[:, 7:].sub_(data[:, 7:].mean())


# Normalize a tree observation
def normalize_observation(tree, max_depth, zero_center=True):
    data = torch.cat([create_tree_features(t, 0, max_depth, ZERO_NODE, []) for t in tree.values()]
                     if isinstance(tree, dict) else
                     create_tree_features(tree, 0, max_depth, ZERO_NODE, []), 0).view((-1, 11))

    wrap(data)

    return data.view(-1)
