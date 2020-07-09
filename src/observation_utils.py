import torch

try:
    from .tree_observation import ACTIONS, negative_infinity, positive_infinity
except:
    from tree_observation import ACTIONS, negative_infinity, positive_infinity

ZERO_NODE = torch.zeros((1, 11))


# Recursively create numpy arrays for each tree node
def create_tree_features(node, max_depth, data):
    nodes = [(node, 0)]
    for node, current_depth in nodes:
        if node == negative_infinity or node == positive_infinity or node is None:
            num_remaining_nodes = (4 ** (max_depth - current_depth + 1) - 1) // 3
            data.append(ZERO_NODE.expand(num_remaining_nodes, -1))
        else:
            data.append(torch.FloatTensor(node[:-2]).view(1, 11))
            if node.childs:
                nodes.extend((node.childs[direction], current_depth + 1) for direction in ACTIONS)


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
def normalize_observation(tree, max_depth, shared_tensor, inner_index):
    data = []
    if isinstance(tree, dict):
        tree = tree.values()
    for t in tree:
        data.append([])
        if isinstance(t, dict):
            any(create_tree_features(d, max_depth, data[-1]) for d in t.values())
        else:
            create_tree_features(t, max_depth, data[-1])
    data = torch.stack([torch.cat(dat, 0) for dat in data], -1)

    wrap(data)

    shared_tensor[inner_index] = data.flatten(0, 1)
