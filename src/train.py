import argparse
import copy
import time
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
# from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from pathos import multiprocessing

# import cv2


torch.jit.optimized_execution(True)

positive_infinity = int(1e5)
negative_infinity = -positive_infinity

try:
    from .rail_env import RailEnv
    from .agent import Agent as DQN_Agent, device, BATCH_SIZE, UPDATE_EVERY
    from .normalize_output_data import wrap
    from .observation_utils import normalize_observation, TreeObservation, GlobalObsForRailEnv, LocalObsForRailEnv, \
        GlobalStateObs
    from .railway_utils import load_precomputed_railways, create_random_railways
except:
    from rail_env import RailEnv
    from agent import Agent as DQN_Agent, device, BATCH_SIZE, UPDATE_EVERY
    from normalize_output_data import wrap
    from observation_utils import normalize_observation, TreeObservation, GlobalObsForRailEnv, LocalObsForRailEnv, \
        GlobalStateObs
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
parser.add_argument("--model-depth", type=int, default=5, help="Depth of the observation tree")
parser.add_argument("--hidden-factor", type=int, default=48, help="Depth of the observation tree")
parser.add_argument("--kernel-size", type=int, default=1, help="Depth of the observation tree")
parser.add_argument("--squeeze-heads", type=int, default=4, help="Depth of the observation tree")
parser.add_argument("--observation-size", type=int, default=4, help="Depth of the observation tree")
parser.add_argument("--decoder-depth", type=int, default=1, help="Depth of the observation tree")

# Training parameters
parser.add_argument("--step-reward", type=float, default=-1e-2, help="Depth of the observation tree")
parser.add_argument("--collision-reward", type=float, default=-2, help="Depth of the observation tree")
parser.add_argument("--global-environment", type=boolean, default=False, help="Depth of the observation tree")
parser.add_argument("--local-environment", type=boolean, default=False, help="Depth of the observation tree")
parser.add_argument("--state-environment", type=boolean, default=True, help="Depth of the observation tree")
parser.add_argument("--threads", type=int, default=1, help="Depth of the observation tree")

flags = parser.parse_args()

if sum((flags.global_environment, flags.local_environment, flags.state_environment)) > 1:
    print("Too many environment flags used. Priority is global > local > state.")

if flags.global_environment:
    model_type = 1
    env = GlobalObsForRailEnv()
elif flags.local_environment:
    model_type = 1
    env = LocalObsForRailEnv(flags.observation_size)
elif flags.state_environment:
    model_type = 2
    env = GlobalStateObs()
else:
    model_type = 0
    env = TreeObservation(flags.tree_depth)

if model_type not in (0, 1, 2):
    raise UserWarning("Unknown model type")

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
                  flags.decoder_depth,
                  model_type)
if flags.load_model:
    start, = agent.load(project_root / 'checkpoints', 0)
else:
    start = 0
# We need to either load in some pre-generated railways from disk, or else create a random railway generator.
rail_generator, schedule_generator = load_precomputed_railways(project_root, start * BATCH_SIZE)

# Create the Flatland environment
environments = [RailEnv(width=50, height=50, number_of_agents=1,
                        rail_generator=rail_generator,
                        schedule_generator=schedule_generator,
                        malfunction_generator_and_process_data=malfunction_from_params(
                            MalfunctionParameters(1 / 500, 20, 50)),
                        obs_builder_object=copy.deepcopy(env),
                        random_seed=i)
                for i in range(BATCH_SIZE)]
env = environments[0]
# After training we want to render the results so we also load a renderer

# Add some variables to keep track of the progress

agent_action_buffer = []
start_time = time.time()

# Helper function to detect collisions
ACTIONS = {0: 'B', 1: 'L', 2: 'F', 3: 'R', 4: 'S'}

if model_type in (1, 2):
    def is_collision(a, i):
        own_agent = environments[i].agents[a]
        return any(own_agent.position == agent.position
                   for agent_id, agent in enumerate(environments[i].agents)
                   if agent_id != a)
else:  # model_type == 0
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


def as_tensor(array_list):
    return torch.as_tensor(np.stack(array_list, 0), dtype=torch.float, device=device)


def make_tensor(current, old):
    if model_type == 1:
        tensor = (torch.cat([old[0], current[0]], -1),)
    elif model_type == 0:
        tensor = (torch.cat([old[0].flatten(1, 2), current[0].flatten(1, 2)], 1),)
    else:  # model_type == 2
        tensor = (torch.cat((old[0], current[0]), 1), torch.cat((old[1], current[1]), -1))
        tensor[1].transpose_(1, -1)
    tensor = tuple(t.to(device) for t in tensor)
    return tensor


def clone(tensor_tuple):
    return tuple(t.clone().detach().requires_grad_(t.requires_grad) for t in tensor_tuple)


def get_observation_tensor(observation, prev_tensor=None):
    buffer = None if prev_tensor is None else clone(prev_tensor)
    if model_type == 1:
        obs_tensor = as_tensor(observation)
    elif model_type == 0:
        obs_tensor = torch.zeros((BATCH_SIZE, state_size // 11, 11, agent_count))
        normalize(observation, obs_tensor)
    else:  # model_type == 2
        rail, obs = zip(*observation)
        obs_tensor = as_tensor(rail), as_tensor(obs)
    return obs_tensor, buffer


episode = 0
POOL = multiprocessing.Pool()
# env_renderer = None
# def render():
#     env_renderer.render_env(show_observations=False)
#     cv2.imshow('Render', cv2.cvtColor(env_renderer.get_image(), cv2.COLOR_BGR2RGB))
#     cv2.waitKey(120)

# Main training loop
episode = start
running_stats = {'score': 0, 'steps': 0, 'collisions': 0, 'done': 0, 'finished': 0}
batch_start = time.time()
while True:
    episode += 1
    agent.reset()
    obs, info = zip(*[env.reset() for env in environments])
    # env_renderer = RenderTool(environments[0], gl="PILSVG", screen_width=1000, screen_height=1000, agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX)
    # env_renderer.reset()
    score, collision = 0, False
    agent_count = len(environments[0].agents)

    agent_action_buffer = [[2] * agent_count for _ in range(BATCH_SIZE)]
    agent_obs, _ = get_observation_tensor(obs)
    agent_obs_buffer = clone(agent_obs)
    input_tensor = make_tensor(agent_obs, agent_obs_buffer)
    # Run an episode
    city_count = (env.width * env.height) // 300
    max_steps = int(8 * (env.width + env.height + agent_count / city_count)) - 10
    # -10 = have some distance to the "real" max steps
    done = [[False]]
    _done = [[False]]
    step = 0
    for step in range(max_steps):
        done = _done

        ret_action = agent.multi_act(input_tensor)
        action_dict = [dict(enumerate(act_list)) for act_list in ret_action]

        # Environment step
        obs, rewards, _done, info = tuple(zip(*[e.step(a) for e, a in zip(environments, action_dict)]))
        score += sum(i for r in rewards for i in r.values()) / (agent_count * BATCH_SIZE)

        # Check for collisions and episode completion
        all_done = (step == (max_steps - 1)) or all(d['__all__'] for d in _done)
        collision = [[is_collision(a, i) for a in range(agent_count)] for i in range(BATCH_SIZE)]
        # Update replay buffer and train agent
        agent_obs, agent_obs_buffer = get_observation_tensor(obs, agent_obs)
        next_input = make_tensor(agent_obs, agent_obs_buffer)

        if flags.train:
            agent.step(input_tensor,
                       agent_action_buffer,
                       _done,
                       collision,
                       next_input,
                       flags.step_reward,
                       flags.collision_reward)
            agent_action_buffer = [[act[i] for i in range(agent_count)] for act in action_dict]

        if all_done:
            break
        input_tensor = next_input

        # Render
        # if flags.render_interval and episode % flags.render_interval == 0:
        # if collision and all(agent.position for agent in env.agents):
        # if step % 2 == 1:
        #    print([a.position for a in environments[0].agents])
        #    render()
        #     print("Collisions detected by agent(s)", ', '.join(str(a) for a in obs if is_collision(a)))
        #     break
    running_stats['score'] += score / max_steps
    running_stats['steps'] += step
    running_stats['collisions'] += sum(i for c in collision for i in c) / agent_count
    running_stats['done'] += sum(d[i] for d in done for i in range(agent_count)) / agent_count
    running_stats['finished'] += sum(d["__all__"] for d in done)

    if episode % UPDATE_EVERY == 0:
        running_stats = {k: v / UPDATE_EVERY for k, v in running_stats.items()}
        print(f'\rBatch{episode:>3} - Episode{BATCH_SIZE * episode:>5} - Agents:{agent_count:>3}'
              f' | Score: {running_stats["score"]:.4f}'
              f' | Steps: {running_stats["steps"]:4.0f}'
              f' | Collisions: {100 * running_stats["collisions"] / BATCH_SIZE:6.2f}%'
              f' | Done: {100 * running_stats["done"] / BATCH_SIZE:6.2f}%'
              f' | Finished: {100 * running_stats["finished"] / BATCH_SIZE:6.2f}%'
              f' | Took: {time.time() - batch_start:5.0f}s')
        running_stats = {k: 0 for k in running_stats.keys()}
        batch_start = time.time()

#    if flags.train:
#        agent.save(project_root / 'checkpoints', episode)
