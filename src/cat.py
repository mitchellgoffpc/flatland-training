import os
import pickle
import random

import tqdm


def main(bucket0, bucket1):
    files = sorted([i for i in os.listdir() if i.endswith('pkl') and 'sum' not in i])
    print(f'Concatenating {", ".join(files)}')

    buckets = [{}, {}]
    for fname in tqdm.tqdm(files, ncols=120, leave=False):
        with open(fname, 'rb') as f:
            try:
                dat = pickle.load(f)
            except Exception as e:
                print(f'Caught {e} while processing {fname}')
            dat = list(dat)
            try:
                _, _ = dat[0]
            except:
                buckets[1][fname.split('_')[-1].split('.pkl')[0]] = dat[:]
            else:
                buckets[0][fname.split('_')[-1].split('.pkl')[0]] = dat[:]

    def _get_itm(idx):
        items = sorted(list(buckets[idx].items()))
        random.seed(0)
        random.shuffle(items)
        names, items = list(zip(*items))
        print(f"First, Last in sequence: {names[0]}, {names[-1]}")
        print(f"Random number _after_ shuffling (to check for seed consitency): {random.random()}")
        buckets[idx] = [itm for lst in items for itm in lst]
        random.shuffle(buckets[idx])

    def _dump(idx, dump_name: str):
        dump_name += 'sum.pkl'
        with open(dump_name, 'wb') as f:
            pickle.dump(buckets[idx], f)
        print(f'Dumped {len(buckets[idx])} items from {len(files)} sources to {dump_name}')

    _get_itm(0)
    _get_itm(1)
    _dump(0, bucket0)
    _dump(1, bucket1)


if __name__ == '__main__':
    main('rail_networks_', 'schedules_')
