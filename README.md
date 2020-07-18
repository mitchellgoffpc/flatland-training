# flatland-training

PyTorch solution for [flatland-2020](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/)

## Implementation

This repository contains three major modules. 

### Getting Started

#### Setup

Before following along here, please note that there is an `install.sh` script which executes all the commands here.\
First, create virtual environment using `python3.7 -m venv venv && source venv/bin/activate`.\
Then install the requirements with `python3 -m pip install -r requirements.txt`.\
It is recommended to verify that the installation was successful by first checking the python version and then attempting an import of all required non-standard packages.
```bash
$ python3 --version
Python 3.7.6
$ python3 -c "import torch, torch_optimizer, numpy, cython, flatland, gym, tqdm; print('Successfully imported packages')"
Successfully imported packages
```
Lastly perhaps the most crucial step. It requires `gcc-7`, as no other version works. On Debian/Ubuntu, it can be installed using the apt package manager by running `apt install gcc-7`.\
Once that's done, the python code can be compiled using cython. Compilation is done by first moving into the source folder and then executing cythonize.sh via `cd src && bash cythonize.sh`.

#### Generate Environments

For better training performance, one can optionally generate the environments used to train the network _before_  training it. This way training is much faster as training data doesn't have to be regenerated repeatedly but instead gets loaded once on startup.\
To start the generation of environments and their respective railways, use `python3 src/generate_railways.py --width 50`. The command will generate 50x50 grids of cities, rails and trains.

#### Run Training

Finally, it's time to train the model. You can do so by running `python3 src/train.py`, which will train a basic cnn using the "local observation" method. ![https://flatland.aicrowd.com/getting-started/env.html](https://i.imgur.com/oo8EIYv.png)\
Not only global and tree observation, but also many model parameters are implemented as well. To find out more about them, add the  `--help` flag.

### Structure

Currently the code is structured in huge monolithic files. 

| Name | Description |
|----|----|
| agent.py | Reward and trainings algorithm, as well as some hyper parameters (such as batch size and learning rate)|
| generate_railways.py | Script to pre-compute and generate railways from the command line. See [#Generate Environments](#Generate-Environments)|
| model.py | PyTorch definition of tree-observation and local-observation models|
| observation_utils.pyx | Agent observation utilities called by environment to create training observation |
| rail_env.pyx | Cython-port of flatland-rl RailEnv |
| railway_utils.py | Utility script to handle creation of and iterators over railways |
| train.py | Core training loop |

### Future Work

The current implementation has many holes. One of them is the very poor performance received when controlling many (>10) agents at once.\
We tackle this issue from multiple sites at once. If you would like to participate in this team, open an issue, pull request or join us on [discord](https://discord.gg/mP72wbE).
Our current approaches are listed below:

* **Observation, Model**:
    * Tree observation, graph neural networks
    * Tree observation, fully-connected networks
    * Tree observation, transformer
    * Local observation, cnn
    * Global observation, cnn 
* **Teaching algorithm**:
    * PPO
    * (Double-) DQN
* **Misc. Freebies**:
    * Epsilon-Greedy
    * Multiprocessing
    * Inter-agent communication
    
If you are working on one of these tasks or would like to do so, please open an issue or pull request to let others know about it. Once seen, it will be added to the main repository.