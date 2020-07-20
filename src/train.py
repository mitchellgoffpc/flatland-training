import argparse
import time
from itertools import zip_longest
from pathlib import Path

import torch
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from pathos import multiprocessing

torch.jit.optimized_execution(True)

positive_infinity = int(1e5)
negative_infinity = -positive_infinity

try:
    from .rail_env import RailEnv
    from .agent import Agent as DQN_Agent, device, BATCH_SIZE
    from .normalize_output_data import wrap
    from .observation_utils import normalize_observation, TreeObservation, GlobalObsForRailEnv, LocalObsForRailEnv
    from .railway_utils import load_precomputed_railways, create_random_railways
except:
    from rail_env import RailEnv
    from agent import Agent as DQN_Agent, device, BATCH_SIZE
    from normalize_output_data import wrap
    from observation_utils import normalize_observation, TreeObservation, GlobalObsForRailEnv, LocalObsForRailEnv
    from railway_utils import load_precomputed_railways, create_random_railways

project_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="Train an agent in the flatland environment")
boolean = lambda x: str(x).lower() == 'true'

# Task parameters
parser.add_argument("--train", type=boolean, default=True, help="Whether to train the model or just evaluate it")
parser.add_argument("--load-model", default=False, action='store_true',
                    help="Whether to load the model from the last checkpoint")
parser.add_argument("--render-interval", type=int, default=0, help="Iterations between renders")

# Environment parameters
parser.add_argument("--tree-depth", type=int, default=2, help="Depth of the observation tree")
parser.add_argument("--model-depth", type=int, default=3, help="Depth of the observation tree")
parser.add_argument("--hidden-factor", type=int, default=48, help="Depth of the observation tree")
parser.add_argument("--kernel-size", type=int, default=1, help="Depth of the observation tree")
parser.add_argument("--squeeze-heads", type=int, default=4, help="Depth of the observation tree")
parser.add_argument("--observation-size", type=int, default=4, help="Depth of the observation tree")

# Training parameters
parser.add_argument("--step-reward", type=float, default=-1e-2, help="Depth of the observation tree")
parser.add_argument("--collision-reward", type=float, default=-2, help="Depth of the observation tree")
parser.add_argument("--global-environment", type=boolean, default=True, help="Depth of the observation tree")
parser.add_argument("--local-environment", type=boolean, default=True, help="Depth of the observation tree")
parser.add_argument("--threads", type=int, default=1, help="Depth of the observation tree")

flags = parser.parse_args()

if flags.local_environment:
    flags.global_environment = True

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
    start, = agent.load(project_root / 'checkpoints', 0)
else:
    start = 0
# We need to either load in some pre-generated railways from disk, or else create a random railway generator.
rail_generator, schedule_generator = load_precomputed_railways(project_root, start * BATCH_SIZE)

# Create the Flatland environment
environments = [RailEnv(width=40, height=40, number_of_agents=1,
                        rail_generator=rail_generator,
                        schedule_generator=schedule_generator,
                        malfunction_generator_and_process_data=malfunction_from_params(
                            MalfunctionParameters(1 / 500, 20, 50)),
                        obs_builder_object=((LocalObsForRailEnv(flags.observation_size)
                                             if flags.local_environment
                                             else GlobalObsForRailEnv)
                                            if flags.global_environment
                                            else TreeObservation(max_depth=flags.tree_depth)),
                        random_seed=i)
                for i in range(BATCH_SIZE)]
env = environments[0]
# After training we want to render the results so we also load a renderer

# Add some variables to keep track of the progress
current_score = current_collisions = current_done = mean_score = mean_collisions = mean_done = current_taken = mean_taken = current_finished = mean_finished = None

agent_action_buffer = []
start_time = time.time()

# Helper function to detect collisions
ACTIONS = {0: 'B', 1: 'L', 2: 'F', 3: 'R', 4: 'S'}

if flags.global_environment:
    def is_collision(a, i):
        own_agent = environments[i].agents[a]
        return any(own_agent.position == agent.position
                   for agent_id, agent in enumerate(environments[i].agents)
                   if agent_id != a)
else:
    def is_collision(a, i):
        if obs[i][a] is None: return False
        is_junction = not isinstance(obs[i][a].childs['L'], float) or not isinstance(obs[i][a].childs['R'], float)

        if not is_junction or environments[i].agents[a].speed_data['position_fraction'] > 0:
            action = ACTIONS[
                environments[i].agents[a].speed_data['transition_action_on_cellexit']] if is_junction else 'F'
            return obs[i][a].childs[action] != negative_infinity and obs[i][a].childs[action] != positive_infinity \
                   and obs[i][a].childs[action].num_agents_opposite_direction > 0 \
                   and obs[i][a].childs[action].dist_other_agent_encountered <= 1 \
                   and obs[i][a].childs[action].dist_other_agent_encountered < obs[i][a].childs[
                       action].dist_unusable_switch
        else:
            return False


def get_means(x, y, c, s):
    return c if x is None else (x * 3 + c) / 4, c if y is None else (y * (s - 1) + c) / s


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
episode = start
while True:
    episode += 1
    agent.reset()
    obs, info = zip(*[env.reset() for env in environments])
    episode_start = time.time()
    score, collision = 0, False
    agent_count = len(obs[0])
    if flags.global_environment:
        agent_obs = torch.as_tensor(obs, dtype=torch.float, device=device)
    else:
        agent_obs = torch.zeros((BATCH_SIZE, state_size // 11, 11, agent_count))
        normalize(obs, agent_obs)

    agent_obs_buffer = agent_obs.clone()
    agent_action_buffer = [[2] * agent_count for _ in range(BATCH_SIZE)]

    # Run an episode
    city_count = (env.width * env.height) // 300
    max_steps = int(8 * (env.width + env.height + agent_count / city_count))
    for step in range(max_steps):
        if flags.global_environment:
            input_tensor = torch.cat([agent_obs_buffer, agent_obs], -1)
            input_tensor.transpose_(1, -1)
        else:
            input_tensor = torch.cat([agent_obs_buffer.flatten(1, 2), agent_obs.flatten(1, 2)], 1)
        ret_action = agent.multi_act(input_tensor)
        action_dict = [dict(enumerate(act_list)) for act_list in ret_action]

        # Environment step
        obs, rewards, done, info = tuple(zip(*[e.step(a) for e, a in zip(environments, action_dict)]))
        score += sum(i for r in rewards for i in r.values()) / (agent_count * BATCH_SIZE)

        # Check for collisions and episode completion
        all_done = (step == (max_steps - 1)) or all(d['__all__'] for d in done)
        collision = [[is_collision(a, i) for a in range(agent_count)] for i in range(BATCH_SIZE)]
        # Update replay buffer and train agent
        if flags.train:
            agent.step(input_tensor,
                       agent_action_buffer,
                       done,
                       collision,
                       flags.step_reward,
                       flags.collision_reward)
            agent_obs_buffer = agent_obs.clone()
            for idx, act in enumerate(action_dict):
                for key, value in act.items():
                    agent_action_buffer[idx][key] = value

        if all_done:
            break

        if flags.global_environment:
            agent_obs = torch.as_tensor(obs, dtype=torch.float, device=device)
        else:
            normalize(obs, agent_obs)

        # Render
        # if flags.render_interval and episode % flags.render_interval == 0:
        # if collision and all(agent.position for agent in env.agents):
        #     render()
        #     print("Collisions detected by agent(s)", ', '.join(str(a) for a in obs if is_collision(a)))
        #     break
    current_collisions, mean_collisions = get_means(current_collisions, mean_collisions,
                                                    sum(i for c in collision for i in c) / (BATCH_SIZE * agent_count),
                                                    episode)
    current_score, mean_score = get_means(current_score, mean_score, score / max_steps, episode)
    current_taken, mean_taken = get_means(current_taken, mean_taken, step, episode)
    current_done, mean_done = get_means(current_done, mean_done,
                                        sum(d[i] for d in done for i in range(agent_count)) / (
                                                BATCH_SIZE * agent_count),
                                        episode)
    current_finished, mean_finished = get_means(current_finished, mean_finished,
                                                sum(d['__all__'] for d in done) / BATCH_SIZE,
                                                episode)

    print(f'\rBatch {episode:>4} - Episode {BATCH_SIZE * episode:>6} - Agents: {agent_count:>3}'
          f' | Score: {current_score:.4f}, {mean_score:.4f}'
          f' | Steps Taken: {current_taken:6.1f}, {mean_taken:6.1f}'
          f' | Collisions: {100 * current_collisions:6.2f}%, {100 * mean_collisions:6.2f}%'
          f' | Agent Done: {100 * current_done:6.2f}%, {100 * mean_done:6.2f}%'
          f' | Finished: {100 * current_finished:6.2f}%, {100 * mean_finished:6.2f}%'
          f' | Episode/s: {BATCH_SIZE * episode / (time.time() - start_time):7.4f}s - Took: {time.time()-episode_start:7.1f}', end='')

    print("")
    if flags.train:
        agent.save(project_root / 'checkpoints', episode)
