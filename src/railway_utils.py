import os
import pickle

import numpy as np

from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

try:
    from .agent import BATCH_SIZE
except:
    from agent import BATCH_SIZE


class Generator:
    def __init__(self, path, start_index=0):
        self.path = path
        self.index = start_index
        self.data = iter([])
        self.len = 0

        self._load()

    def _load(self):
        with open(self.path, 'rb') as file:
            data = pickle.load(file)
            self.len = len(data)
            self.data = iter(data[self.index:])

    def __next__(self):
        try:
            data = next(self.data)
        except StopIteration:
            self._load()
            if self.index >= self.len:
                print("[WARNING] Restarting training loop from zero")
                self.index = 0
                self._load()
            data = next(self)
        self.index += 1
        return data

    def __len__(self):
        return self.len

    def __call__(self, *args, **kwargs):
        return next(self)


class RailGenerator:
    def __init__(self, width=35, base=1.5):
        self.rail_generator = sparse_rail_generator(grid_mode=False,
                                                    max_num_cities=max(2, width ** 2 // 300),
                                                    max_rails_between_cities=2,
                                                    max_rails_in_city=3)
        self.sub_idx = 0
        self.top_idx = 0
        self.width = width
        self.base = base

    def __next__(self):
        self.sub_idx += 1
        if self.sub_idx == BATCH_SIZE:
            self.sub_idx = 0
            self.top_idx += 1
        return self.rail_generator(self.width, self.width, int(2 * self.base ** self.top_idx), np_random=np.random)

    def __call__(self, *args, **kwargs):
        return next(self)


class ScheduleGenerator:
    def __init__(self, base=1.5):
        self.schedule_generator = sparse_schedule_generator({1.: 1.})
        self.sub_idx = 0
        self.top_idx = 0
        self.base = base

    def __next__(self, rail, hints):
        if self.sub_idx == BATCH_SIZE:
            self.sub_idx = 0
            self.top_idx += 1
        self.sub_idx += 1
        return self.schedule_generator(rail, int(2 * self.base ** self.top_idx), hints, np_random=np.random)

    def __call__(self, rail, _, hints, *args, **kwargs):
        return self.__next__(rail, hints)


# Helper function to load in precomputed railway networks
def load_precomputed_railways(project_root, start_index, big=True):
    prefix = os.path.join(project_root, 'railroads')
    if big:
        suffix = f'_45x90x90.pkl'
    else:
        suffix = f'_3x30x30.pkl'
    sched = Generator(os.path.join(prefix, 'rail_networks' + suffix), start_index)
    rail = Generator(os.path.join(prefix, 'schedules' + suffix), start_index)
    #if big:
    #    sched, rail = rail, sched
    print(f"Working on {len(rail)} tracks")
    return rail, sched


# Helper function to generate railways on the fly
def create_random_railways(width, base=1.1):
    return RailGenerator(width=width, base=base), ScheduleGenerator(base=base)
