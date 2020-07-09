import numpy as np
import torch

try:
    from .tree_observation import ACTIONS
except:
    from tree_observation import ACTIONS

ZERO_NODE = np.array([0] * 11)  # For Q-Networks
INF_DISTANCE_NODE = np.array([0] * 6 + [np.inf] + [0] * 4)  # For policy networks


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
        data.append(np.array(tuple(node)[:-2]))
        if node.childs:
            for direction in ACTIONS:
                create_tree_features(node.childs[direction], current_depth + 1, max_depth, empty_node, data)

    return data


TRUE = torch.ones(1)
FALSE = torch.zeros(1)


# Normalize an observation to [0, 1] and then clip it to get rid of any infinite-valued features
#@torch.jit.script
def norm_obs_clip(obs, normalize_to_range):
    max_obs = obs[obs < 1000].max()
    max_obs.clamp_(min=1)
    max_obs.add_(1)

    min_obs = torch.zeros(1)[0]

    if normalize_to_range.item():
        min_obs.add_(obs[obs >= 0].min().clamp(max=max_obs.item()))

    if max_obs == min_obs:
        obs.div_(max_obs)
    else:
        obs.sub_(min_obs)
        max_obs.sub_(min_obs)
        obs.div_(max_obs)
    return obs


# Normalize a tree observation
def normalize_observation(tree, max_depth, zero_center=True):
    empty_node = ZERO_NODE if zero_center else INF_DISTANCE_NODE
    data = np.concatenate([create_tree_features(t, 0, max_depth, empty_node, []) for t in tree.values()]
                          if isinstance(tree, dict) else
                          create_tree_features(tree, 0, max_depth, empty_node, [])).reshape((-1, 11))
    data = torch.as_tensor(data).float()

    norm_obs_clip(data[:, :6], FALSE)
    norm_obs_clip(data[:, 6], TRUE)
    data.clamp_(-1, 1)

    if zero_center:
        data[:, :6].sub_(data[:, :6].mean())
        data[:, 7:].sub_(data[:, 7:].mean())
    return data.flatten()
