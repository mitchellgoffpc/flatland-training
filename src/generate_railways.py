import time
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, complex_schedule_generator

from railway_utils import create_random_railways


project_root = Path(__file__).resolve().parent.parent
rail_generator, schedule_generator = create_random_railways(project_root)

width, height = 50, 50
n_agents = 5


# Load in any existing railways for this map size so we don't overwrite them
try:
    with open(project_root / f'railroads/rail_networks_{n_agents}x{width}x{height}.pkl', 'rb') as file:
        rail_networks = pickle.load(file)
    with open(project_root / f'railroads/schedules_{n_agents}x{width}x{height}.pkl', 'rb') as file:
        schedules = pickle.load(file)
    print(f"Loading {len(rail_networks)} railways...")
except:
    rail_networks, schedules = [], []


# Generate 10000 random railways in 100 batches of 100
for _ in range(100):
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
