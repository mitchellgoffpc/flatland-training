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

parser.add_argument("--width", type=int, default=35, help="Decay factor for epsilon-greedy exploration")
parser.add_argument("--factor", type=int, default=2, help="Decay factor for epsilon-greedy exploration")
parser.add_argument("--base", type=float, default=1.1, help="Decay factor for epsilon-greedy exploration")
flags = parser.parse_args()

rail_generator, schedule_generator = create_random_railways(flags.width, flags.base, flags.factor)

# Load in any existing railways for this map size so we don't overwrite them
network = project_root / f'railroads/rail_networks_{flags.width}_{flags.factor}.pkl'
sched = project_root / f'railroads/schedules_{flags.width}_{flags.factor}.pkl'
try:
    with open(network, 'rb') as file:
        rail_networks = pickle.load(file)
    with open(sched, 'rb') as file:
        schedules = pickle.load(file)
    print(f"Loading {len(rail_networks)} railways...")
except:
    rail_networks, schedules = [], []


def do(schedules: list, rail_networks: list):
    for _ in range(100):
        map, info = rail_generator(flags.width, 1, 1, num_resets=0, np_random=np.random)
        schedule = schedule_generator(map, 1, info['agents_hints'], num_resets=0, np_random=np.random)
        rail_networks.append((map, info))
        schedules.append(schedule)
    return


for _ in tqdm(range(500), ncols=150, leave=False):
    do(schedules, rail_networks)
    with open(network, 'wb') as file:
        pickle.dump(rail_networks, file, protocol=4)
    with open(sched, 'wb') as file:
        pickle.dump(schedules, file, protocol=4)

print(f"Saved {len(rail_networks)} railways")
print("Done")
