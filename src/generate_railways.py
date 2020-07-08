import argparse
import multiprocessing
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from .railway_utils import create_random_railways
except:
    from railway_utils import create_random_railways

project_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="Train an agent in the flatland environment")

parser.add_argument("--agents", type=int, default=3, help="Number of episodes to train for")
parser.add_argument("--cities", type=int, default=3, help="Number of episodes to train for")
parser.add_argument("--width", type=int, default=35, help="Decay factor for epsilon-greedy exploration")

flags = parser.parse_args()

width = height = flags.width
n_agents = flags.agents
rail_generator, schedule_generator = create_random_railways(project_root, flags.cities)

# Load in any existing railways for this map size so we don't overwrite them
try:
    with open(project_root / f'railroads/rail_networks_{n_agents}x{width}x{height}.pkl', 'rb') as file:
        rail_networks = pickle.load(file)
    with open(project_root / f'railroads/schedules_{n_agents}x{width}x{height}.pkl', 'rb') as file:
        schedules = pickle.load(file)
    print(f"Loading {len(rail_networks)} railways...")
except:
    rail_networks, schedules = [], []


def do(schedules: list, rail_networks: list):
    for _ in range(100):
        map, info = rail_generator(width, height, n_agents, num_resets=0, np_random=np.random)
        schedule = schedule_generator(map, n_agents, info['agents_hints'], num_resets=0, np_random=np.random)
        rail_networks.append((map, info))
        schedules.append(schedule)
    return


manager = multiprocessing.Manager()
shared_schedules = manager.list(schedules)
shared_rail_networks = manager.list(rail_networks)
# Generate 10000 random railways in 100 batches of 100
for _ in tqdm(range(100), ncols=120, leave=False):
    do(schedules, rail_networks)
    with open(project_root / f'railroads/rail_networks_{n_agents}x{width}x{height}.pkl', 'wb') as file:
        pickle.dump(schedules, file, protocol=4)
    with open(project_root / f'railroads/schedules_{n_agents}x{width}x{height}.pkl', 'wb') as file:
        pickle.dump(rail_networks, file, protocol=4)

print(f"Saved {len(shared_rail_networks)} railways")
print("Done")
