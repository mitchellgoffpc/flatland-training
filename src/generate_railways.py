import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, complex_schedule_generator

project_root = Path(__file__).parent.parent

speed_ration_map = {
    1.: 0.,        # Fast passenger train
    1. / 2.: 1.0,   # Fast freight train
    1. / 3.: 0.0,   # Slow commuter train
    1. / 4.: 0.0 }  # Slow freight train

rail_generator = sparse_rail_generator(grid_mode=False, max_num_cities=4, max_rails_between_cities=3, max_rails_in_city=4, seed=1)
schedule_generator = sparse_schedule_generator(speed_ration_map)

# rail_generator = complex_rail_generator(nr_start_goal=5, nr_extra=5, min_dist=10, max_dist=99999)
# schedule_generator = complex_schedule_generator()

width, height = 35, 35
n_agents = 1

try:
    with open(project_root / f'railroads/rail_networks_{n_agents}x{width}x{height}.pkl', 'rb') as file:
        rail_networks = pickle.load(file)
    with open(project_root / f'railroads/schedules_{n_agents}x{width}x{height}.pkl', 'rb') as file:
        schedules = pickle.load(file)
except:
    rail_networks, schedules = [], []

print(f"Loading {len(rail_networks)} railways")

for _ in range(50): # generate 5000 episodes
    for i in tqdm(range(100), ncols=120, leave=False):
        map, info = rail_generator(width, height, n_agents, num_resets=0, np_random=np.random)
        schedule = schedule_generator(map, n_agents, info['agents_hints'], num_resets=0, np_random=np.random)
        rail_networks.append((map, info))
        schedules.append(schedule)

    print(f"Saving {len(rail_networks)} railways")
    with open(project_root / f'railroads/rail_networks_{n_agents}x{width}x{height}.pkl', 'wb') as file:
        pickle.dump(rail_networks, file)
    with open(project_root / f'railroads/schedules_{n_agents}x{width}x{height}.pkl', 'wb') as file:
        pickle.dump(schedules, file)

print("Done")
