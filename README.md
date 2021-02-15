# flatland-training

This repo contains an optimized version of flatland-rl's `flatland.envs.observations.TreeObsForRailEnv`. Tree-based observations allow RL models to learn much more quickly than the global observations do, but flatland's built-in TreeObsForRailEnv is kind of slow, so I wrote a faster version! This repo also contains an optimized version of [https://gitlab.aicrowd.com/flatland/baselines/blob/master/utils/observation_utils.py](https://gitlab.aicrowd.com/flatland/baselines/blob/master/utils/observation_utils.py), which flattens and normalizes the tree observations into 1D numpy arrays that can be passed to a feed-forward network.


## Setup
`pip install -r requirements.txt`


## Generate Railways
This script will precompute a bunch of railway maps to make training faster.

`python src/generate_railways.py`

This will run for quite a long time, so go get some tea! If you don't care about the speedup, you can run `python src/train.py --load-railways=False` to generate railways on the fly during training instead.


## Run Training
`python src/train.py`

This will begin training one or more agents in the flatland environment. This file has a lot of parameters that can be set to do different things. To see all the options, use the `--help` command line argument.
