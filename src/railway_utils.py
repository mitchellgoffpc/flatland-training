import os
import pickle

from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator


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


# Helper function to load in precomputed railway networks
def load_precomputed_railways(project_root, start_index, big=False):
    prefix = os.path.join(project_root, 'railroads')
    if big:
        suffix = f'_50x35x20.pkl'
    else:
        suffix = f'_3x30x30.pkl'
    sched = Generator(os.path.join(prefix, 'rail_networks' + suffix), start_index)
    rail = Generator(os.path.join(prefix, 'schedules' + suffix), start_index)
    if big:
        sched, rail = rail, sched
    print(f"Working on {len(rail)} tracks")
    return rail, sched


# Helper function to generate railways on the fly
def create_random_railways(project_root, max_cities=5):
    speed_ratio_map = {
        1 / 1: 1.0,  # Fast passenger train
        1 / 2.: 0.0,  # Fast freight train
        1 / 3.: 0.0,  # Slow commuter train
        1 / 4.: 0.0}  # Slow freight train

    rail_generator = sparse_rail_generator(grid_mode=False, max_num_cities=max_cities,
                                           max_rails_between_cities=2, max_rails_in_city=3)
    schedule_generator = sparse_schedule_generator(speed_ratio_map)
    return rail_generator, schedule_generator
