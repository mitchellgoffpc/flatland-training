import torch

try:
    from .tree_observation import ACTIONS, negative_infinity, positive_infinity
except:
    from tree_observation import ACTIONS, negative_infinity, positive_infinity


ZERO_NODE = torch.zeros((1, 11))

# Recursively create numpy arrays for each tree node
cpdef create_tree_features(node, int max_depth, list data):
    cdef list nodes = [(node, 0)]
    cdef int current_depth = 0
    for node, current_depth in nodes:
        if node == negative_infinity or node == positive_infinity or node is None:
            data.append(ZERO_NODE.expand((4 ** (max_depth - current_depth + 1) - 1) // 3, -1))
        else:
            data.append(torch.FloatTensor(node[:-2]).view(1, 11))
            if node.childs:
                for direction in ACTIONS:
                    nodes.append((node.childs[direction], current_depth + 1))

# Normalize a tree observation
cpdef normalize_observation(tuple observations, int max_depth, shared_tensor, int starting_index):
    cdef list data = []
    cdef int i = 0
    for i, tree in enumerate(observations, 1):
        if tree is None:
            break
        data.append([])
        if isinstance(tree, dict):
            tree = tree.values()
        for t in tree:
            data[-1].append([])
            if isinstance(t, dict):
                for d in t.values():
                    create_tree_features(d, max_depth, data[-1][-1])
            else:
                create_tree_features(t, max_depth, data[-1][-1])

    shared_tensor[starting_index:starting_index + i] = torch.stack([torch.stack([torch.cat(dat, 0)
                                                                                 for dat in tree if dat != []], -1)
                                                                    for tree in data if tree != []], 0)