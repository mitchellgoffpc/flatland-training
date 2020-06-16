import numpy as np
from tree_observation import ACTIONS

EMPTY_NODE = np.array([-np.inf] * 11)


def max_lt(seq, val):
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min


def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1
        # max_obs = max(1, min(1000, *obs)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
        # min_obs = max(0, *obs)
    if min_obs > max_obs:
        min_obs = max_obs

    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    else:
        norm = np.abs(max_obs - min_obs)
        return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def get_node_features(node):
    data = np.zeros(11)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    data[6] = node.dist_min_to_target

    data[7] = node.num_agents_same_direction
    data[8] = node.num_agents_opposite_direction
    data[9] = node.num_agents_malfunctioning
    data[10] = node.speed_min_fractional

    return data


def get_tree_features(node, depth, max_depth, data):
    if node == -np.inf:
        num_remaining_nodes = (4 ** (max_depth - depth + 1) - 1) // (4 - 1)
        data.extend([EMPTY_NODE] * num_remaining_nodes)

    else:
        data.append(get_node_features(node))
        if node.childs:
            for direction in ACTIONS:
                get_tree_features(node.childs[direction], depth + 1, max_depth, data)

    return data


def normalize_observation(tree, max_depth, observation_radius=0):
    data = np.concatenate(get_tree_features(tree, 0, max_depth, []))
    data = data.reshape((-1, 11))

    obs_data = norm_obs_clip(data[:,:6].flatten(), fixed_radius=observation_radius)
    distances = norm_obs_clip(data[:,6], normalize_to_range=True)
    agent_data = np.clip(data[:,7:].flatten(), -1, 1)
    return np.concatenate((obs_data, distances, agent_data))
