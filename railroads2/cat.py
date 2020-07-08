import os
import pickle
import random


def main(base, out_name):
        files = sorted([i for i in os.listdir() if i.startswith(base) and not i.endswith('.bak') and 'sum' not in i])
        print(f'Concatenating {", ".join(files)}')

        out = []

        for name in files:
                with open(name, 'rb') as f:
                        try:
                                out.extend(pickle.load(f))
                        except Exception as e:
                                print(f'Caught {e} while processing {name}')

        name = out_name+'sum.pkl'
        random.seed(0)
        random.shuffle(out)
        with open(name, 'wb') as f:
                pickle.dump(out, f)
        print(f'Dumped {len(out)} items from {len(files)} sources to {name}')

if __name__ == '__main__':
        main('rail_networks_', 'schedules_')
        main('schedules_', 'rail_networks_')

