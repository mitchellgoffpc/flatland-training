import argparse
import copy
import time
from itertools import zip_longest
from pathlib import Path

import torch
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from pathos import multiprocessing

try:
    from .agent import Agent as DQN_Agent, device, BATCH_SIZE
    from .normalize_output_data import wrap
    from .tree_observation import TreeObservation, negative_infinity, positive_infinity
    from .observation_utils import normalize_observation
    from .railway_utils import load_precomputed_railways, create_random_railways
except:
    from agent import Agent as DQN_Agent, device, BATCH_SIZE
    from normalize_output_data import wrap
    from tree_observation import TreeObservation, negative_infinity, positive_infinity
    from observation_utils import normalize_observation
    from railway_utils import load_precomputed_railways, create_random_railways

project_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="Train an agent in the flatland environment")
boolean = lambda x: str(x).lower() == 'true'

# Task parameters
parser.add_argument("--train", type=boolean, default=True, help="Whether to train the model or just evaluate it")
parser.add_argument("--load-model", default=False, action='store_true',
                    help="Whether to load the model from the last checkpoint")
parser.add_argument("--load-railways", type=boolean, default=True,
                    help="Whether to load in pre-generated railway networks")
parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")
parser.add_argument("--render-interval", type=int, default=0, help="Iterations between renders")

# Environment parameters
parser.add_argument("--grid-width", type=int, default=50, help="Number of columns in the environment grid")
parser.add_argument("--grid-height", type=int, default=50, help="Number of rows in the environment grid")
parser.add_argument("--num-agents", type=int, default=5, help="Number of agents in each episode")
parser.add_argument("--tree-depth", type=int, default=1, help="Depth of the observation tree")
parser.add_argument("--model-depth", type=int, default=1, help="Depth of the observation tree")
parser.add_argument("--hidden-factor", type=int, default=5, help="Depth of the observation tree")
parser.add_argument("--kernel-size", type=int, default=1, help="Depth of the observation tree")
parser.add_argument("--squeeze-heads", type=int, default=4, help="Depth of the observation tree")

# Training parameters
parser.add_argument("--agent-type", default="dqn", choices=["dqn", "ppo"], help="Which type of RL agent to use")
parser.add_argument("--num-episodes", type=int, default=10 ** 6, help="Number of episodes to train for")
parser.add_argument("--epsilon-decay", type=float, default=0, help="Decay factor for epsilon-greedy exploration")
parser.add_argument("--step-reward", type=float, default=-1, help="Depth of the observation tree")
parser.add_argument("--global-environment", type=boolean, default=False, help="Depth of the observation tree")
parser.add_argument("--threads", type=int, default=1, help="Depth of the observation tree")

flags = parser.parse_args()

# Seeded RNG so we can replicate our results

# Create a tensorboard SummaryWriter
# Calculate the state size based on the number of nodes in the tree observation
num_features_per_node = 11  # env.obs_builder.observation_dim
num_nodes = int('1' * (flags.tree_depth + 1), 4)
state_size = num_nodes * num_features_per_node
action_size = 5
# Load an RL agent and initialize it from checkpoint if necessary
agent = DQN_Agent(state_size,
                  action_size,
                  flags.model_depth,
                  flags.hidden_factor,
                  flags.kernel_size,
                  flags.squeeze_heads,
                  flags.global_environment)
if flags.load_model:
    start, eps = agent.load(project_root / 'checkpoints', 0, 1.0)
else:
    start, eps = 0, 1.0
# We need to either load in some pre-generated railways from disk, or else create a random railway generator.
if flags.load_railways:
    rail_generator, schedule_generator = load_precomputed_railways(project_root, start)
else:
    rail_generator, schedule_generator = create_random_railways(project_root)

# Create the Flatland environment
env = RailEnv(width=flags.grid_width, height=flags.grid_height, number_of_agents=flags.num_agents,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              malfunction_generator_and_process_data=malfunction_from_params(MalfunctionParameters(1 / 8000, 15, 50)),
              obs_builder_object=(GlobalObsForRailEnv()
                                  if flags.global_environment
                                  else TreeObservation(max_depth=flags.tree_depth))
              )
environments = [copy.copy(env) for _ in range(BATCH_SIZE)]

# After training we want to render the results so we also load a renderer

# Add some variables to keep track of the progress
current_score = current_steps = current_collisions = current_done = mean_score = mean_steps = mean_collisions = mean_done = current_taken = mean_taken = 0

agent_action_buffer = []
start_time = time.time()

if not flags.train:
    eps = 0.0

# Helper function to detect collisions
ACTIONS = {0: 'B', 1: 'L', 2: 'F', 3: 'R', 4: 'S'}


def is_collision(a, i):
    if obs[i][a] is None: return False
    is_junction = not isinstance(obs[i][a].childs['L'], float) or not isinstance(obs[i][a].childs['R'], float)

    if not is_junction or environments[i].agents[a].speed_data['position_fraction'] > 0:
        action = ACTIONS[environments[i].agents[a].speed_data['transition_action_on_cellexit']] if is_junction else 'F'
        return obs[i][a].childs[action] != negative_infinity and obs[i][a].childs[action] != positive_infinity \
               and obs[i][a].childs[action].num_agents_opposite_direction > 0 \
               and obs[i][a].childs[action].dist_other_agent_encountered <= 1 \
               and obs[i][a].childs[action].dist_other_agent_encountered < obs[i][a].childs[action].dist_unusable_switch
    else:
        return False


def get_means(x, y, c, s):
    return (x * 3 + c) / 4, (y * (s - 1) + c) / s


chunk_size = (BATCH_SIZE + 1) // flags.threads


def chunk(obj, size):
    return zip_longest(*[iter(obj)] * size, fillvalue=None)


if flags.threads > 1:
    def normalize(observation, target_tensor):
        POOL.starmap(func=normalize_observation,
                     iterable=((o, flags.tree_depth, target_tensor, i * chunk_size)
                               for i, o in enumerate(chunk(observation, chunk_size))))
        wrap(target_tensor)
else:
    def normalize(observation, target_tensor):
        normalize_observation(observation, flags.tree_depth, target_tensor, 0)
        wrap(target_tensor)

episode = 0
POOL = multiprocessing.Pool()

# Main training loop
for episode in range(start + 1, flags.num_episodes + 1):
    agent.reset()
    obs, info = env.reset(True, True)
    environments = [copy.copy(env) for _ in range(BATCH_SIZE)]
    obs = tuple(copy.deepcopy(obs) for _ in range(BATCH_SIZE))
    info = [copy.deepcopy(info) for _ in range(BATCH_SIZE)]
    score, steps_taken, collision = 0, 0, False
    agent_count = len(obs[0])
    agent_obs = torch.zeros((BATCH_SIZE, state_size // 11, 11, agent_count))
    normalize(obs, agent_obs)
    agent_obs_buffer = agent_obs.clone()
    agent_action_buffer = [[2] * agent_count for _ in range(BATCH_SIZE)]

    # Run an episode
    max_steps = 8 * env.width + env.height
    for step in range(max_steps):
        update_values = [[False] * agent_count for _ in range(BATCH_SIZE)]
        action_dict = [{} for _ in range(BATCH_SIZE)]

        if all(any(inf['action_required']) for inf in info):
            ret_action = agent.multi_act(agent_obs.flatten(1, 2), eps=eps)
        else:
            ret_action = update_values
        for idx, act_list in enumerate(ret_action):
            for sub_idx, act in enumerate(act_list):
                if info[idx]['action_required'][sub_idx]:
                    action_dict[idx][sub_idx] = act
                    # action_dict[a] = np.random.randint(5)
                    update_values[idx][sub_idx] = True
                    steps_taken += 1
                else:
                    action_dict[idx][sub_idx] = 0

        # Environment step
        obs, rewards, done, info = tuple(zip(*[e.step(a) for e, a in zip(environments, action_dict)]))
        score += sum(sum(r.values()) for r in rewards) / (agent_count * BATCH_SIZE)

        # Check for collisions and episode completion
        all_done = (step == (max_steps - 1)) or any(d['__all__'] for d in done)
        if any(is_collision(a, i) for i, o in enumerate(obs) for a in o):
            collision = True
            # done['__all__'] = True

        # Update replay buffer and train agent
        if flags.train and (any(update_values) or all_done or all(any(d) for d in done)):
            agent.step(agent_obs_buffer.flatten(1, 2),
                       agent_action_buffer,
                       agent_obs.flatten(1, 2),
                       done,
                       all_done,
                       [[is_collision(a, i) for a in range(agent_count)] for i in range(BATCH_SIZE)],
                       flags.step_reward)
            agent_obs_buffer = agent_obs.clone()
            for idx, act in enumerate(action_dict):
                for key, value in act.items():
                    agent_action_buffer[idx][key] = value

        if all_done:
            break

        normalize(obs, agent_obs)

        # Render
        # if flags.render_interval and episode % flags.render_interval == 0:
        # if collision and all(agent.position for agent in env.agents):
        #     render()
        #     print("Collisions detected by agent(s)", ', '.join(str(a) for a in obs if is_collision(a)))
        #     break

    # Epsilon decay
    if flags.train:
        eps = max(0.01, flags.epsilon_decay * eps)

    current_collisions, mean_collisions = get_means(current_collisions, mean_collisions, int(collision), episode)
    current_score, mean_score = get_means(current_score, mean_score, score / max_steps, episode)
    current_steps, mean_steps = get_means(current_steps, mean_steps, steps_taken / BATCH_SIZE / agent_count, episode)
    current_taken, mean_taken = get_means(current_steps, mean_steps, step, episode)

    print(f'\rBatch {episode:<5} - Episode {BATCH_SIZE*episode:<5}'
          f' | Score: {current_score:.4f}, {mean_score:.4f}'
          f' | Agent-Steps: {current_steps:6.1f}, {mean_steps:6.1f}'
          f' | Steps Taken: {current_taken:6.1f}, {mean_taken:6.1f}'
          f' | Collisions: {100 * current_collisions:5.2f}%, {100 * mean_collisions:5.2f}%'
          f' | Epsilon: {eps:.2f}'
          f' | Episode/s: {BATCH_SIZE * episode / (time.time() - start_time):.4f}s', end='')

    if episode % flags.report_interval == 0:
        print("")
        if flags.train:
            agent.save(project_root / 'checkpoints', episode, eps)
        # Add stats to the tensorboard summary
