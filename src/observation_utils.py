import numpy as np
from dataclasses import astuple
from tree_observation import ACTIONS

ZERO_NODE = np.array([0] * 11) # For Q-Networks
INF_DISTANCE_NODE = np.array([0] * 6 + [np.inf] + [0] * 4) # For policy networks


# Helper function to detect collisions
def is_collision(obs):
    return obs is not None \
       and isinstance(obs.children['L'], float) \
       and isinstance(obs.children['R'], float) \
       and obs.children['F'].num_agents_opposite_direction > 0 \
       and obs.children['F'].dist_other_agent_encountered <= 1 \
       and obs.children['F'].dist_other_agent_encountered < obs.children['F'].dist_unusable_switch
       # and obs.children['F'].dist_other_agent_encountered < obs.children['F'].dist_to_next_branch


# Recursively create numpy arrays for each tree node
def create_tree_features(node, current_depth, max_depth, empty_node, data):
    if node == -np.inf:
        num_remaining_nodes = (4 ** (max_depth - current_depth + 1) - 1) // (4 - 1)
        data.extend([empty_node] * num_remaining_nodes)

    else:
        data.append(np.array(astuple(node)[:-2]))
        if node.children:
            for direction in ACTIONS:
                create_tree_features(node.children[direction], current_depth + 1, max_depth, empty_node, data)

    return data

# Normalize an observation to [0, 1] and then clip it to get rid of any infinite-valued features
def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    if fixed_radius > 0:
          max_obs = fixed_radius
    else: max_obs = np.max(obs[np.where(obs < 1000)], initial=1) + 1

    min_obs = np.min(obs[np.where(obs >= 0)], initial=max_obs) if normalize_to_range else 0

    if max_obs == min_obs:
          return np.clip(obs / max_obs, clip_min, clip_max)
    else: return np.clip((obs - min_obs) / np.abs(max_obs - min_obs), clip_min, clip_max)


# Normalize a tree observation
def normalize_observation(tree, max_depth, zero_center=True):
    empty_node = ZERO_NODE if zero_center else INF_DISTANCE_NODE
    data = np.concatenate(create_tree_features(tree, 0, max_depth, empty_node, [])).reshape((-1, 11))

    obs_data = norm_obs_clip(data[:,:6].flatten())
    distances = norm_obs_clip(data[:,6], normalize_to_range=True)
    agent_data = np.clip(data[:,7:].flatten(), -1, 1)

    if zero_center:
          return np.concatenate((obs_data - obs_data.mean(), distances, agent_data - agent_data.mean()))
    else: return np.concatenate((obs_data, distances, agent_data))
