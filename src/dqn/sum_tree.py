# This is a binary tree data structure where the parent’s value is the sum of its children.
# Copied from https://github.com/rlcode/per/blob/master/SumTree.py

import numpy as np


class SumTree:
    write = 0
    n_entries = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    # update to the root node
    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self.propagate(parent, change)

    # find sample on leaf node
    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
              return idx
        elif s <= self.tree[left]:
              return self.retrieve(left, s)
        else: return self.retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self.propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self.retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
