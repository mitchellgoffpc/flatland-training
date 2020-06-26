# flatland-training

This repo contains an optimized version of flatland-rl's `flatland.envs.observations.TreeObsForRailEnv`. Tree-based observations allow RL models to learn much more quickly than the global observations do, but flatland's built-in TreeObsForRailEnv is kind of slow, so I wrote a faster version! This repo also contains an optimized version of [https://gitlab.aicrowd.com/flatland/baselines/blob/master/utils/observation_utils.py](https://gitlab.aicrowd.com/flatland/baselines/blob/master/utils/observation_utils.py), which flattens and normalizes the tree observations into 1D numpy arrays that can be passed to a feed-forward network.


# Setup
## Create venv
`python3.7 -m venv venv`\
`source venv/bin/activate`

Verify python version is correct with: `python -V`\
Should return `Python 3.7.something`

## Install Requirements
`pip install --upgrade pip`\
`pip install -r requirements`\

# Generate Railways
This will build railway maps for the agents to run and train within

`python src/generate_railways.py`

This will run for quite a long time, go get some tea...\
But also it's fine to stop it after it completes at least one round if you just want to test things out and make sure they run.

# Run Training
`python src/train.py`

This will begin training one or more agents to solve the problem.\
This file has a lot of parameters that can be set to do different things, go check out the file.