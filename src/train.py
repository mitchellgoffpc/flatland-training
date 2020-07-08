import argparse
import copy
import time
from pathlib import Path

import cv2
import numpy as np
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from tensorboardX import SummaryWriter

try:
    from .agent import Agent as DQN_Agent, device, BATCH_SIZE
    from .tree_observation import TreeObservation
    from .observation_utils import normalize_observation, is_collision
    from .railway_utils import load_precomputed_railways, create_random_railways
except:
    from agent import Agent as DQN_Agent, device, BATCH_SIZE
    from tree_observation import TreeObservation
    from observation_utils import normalize_observation, is_collision
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
parser.add_argument("--model-depth", type=int, default=4, help="Depth of the observation tree")
parser.add_argument("--hidden-factor", type=int, default=15, help="Depth of the observation tree")
parser.add_argument("--kernel-size", type=int, default=1, help="Depth of the observation tree")
parser.add_argument("--squeeze-heads", type=int, default=4, help="Depth of the observation tree")

# Training parameters
parser.add_argument("--agent-type", default="dqn", choices=["dqn", "ppo"], help="Which type of RL agent to use")
parser.add_argument("--num-episodes", type=int, default=10 ** 6, help="Number of episodes to train for")
parser.add_argument("--epsilon-decay", type=float, default=0, help="Decay factor for epsilon-greedy exploration")
parser.add_argument("--step-reward", type=float, default=-1e-2, help="Depth of the observation tree")

flags = parser.parse_args()

# Seeded RNG so we can replicate our results
np.random.seed(1)

# Create a tensorboard SummaryWriter
summary = SummaryWriter(f'tensorboard/dqn/agents: {flags.num_agents}, tree_depth: {flags.tree_depth}')
# Calculate the state size based on the number of nodes in the tree observation
num_features_per_node = 11  # env.obs_builder.observation_dim
num_nodes = sum(np.power(4, i) for i in range(flags.tree_depth + 1))
state_size = num_nodes * num_features_per_node
action_size = 5
# Load an RL agent and initialize it from checkpoint if necessary
agent = DQN_Agent(state_size,
                  action_size,
                  flags.num_agents,
                  flags.model_depth,
                  flags.hidden_factor,
                  flags.kernel_size,
                  flags.squeeze_heads)
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
              obs_builder_object=TreeObservation(max_depth=flags.tree_depth)
              )

# After training we want to render the results so we also load a renderer
env_renderer = RenderTool(env, gl="PILSVG", screen_width=800, screen_height=800,
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX)

# Add some variables to keep track of the progress
current_score = current_steps = current_collisions = current_done = mean_score = mean_steps = mean_collisions = mean_done = 0

agent_action_buffer = []
start_time = time.time()

if not flags.train:
    eps = 0.0

# Helper function to detect collisions
ACTIONS = {0: 'B', 1: 'L', 2: 'F', 3: 'R', 4: 'S'}


def is_collision(a):
    if obs[a] is None: return False
    is_junction = not isinstance(obs[a].childs['L'], float) or not isinstance(obs[a].childs['R'], float)

    if not is_junction or env.agents[a].speed_data['position_fraction'] > 0:
        action = ACTIONS[env.agents[a].speed_data['transition_action_on_cellexit']] if is_junction else 'F'
        return obs[a].childs[action].num_agents_opposite_direction > 0 \
               and obs[a].childs[action].dist_other_agent_encountered <= 1 \
               and obs[a].childs[action].dist_other_agent_encountered < obs[a].childs[action].dist_unusable_switch
    else:
        return False


# Helper function to render the environment
def render():
    env_renderer.render_env(show_observations=False)
    cv2.imshow('Render', cv2.cvtColor(env_renderer.get_image(), cv2.COLOR_BGR2RGB))
    cv2.waitKey(120)


def get_means(x, y, c, s):
    return (x * 3 + c) / 4, (y * (s - 1) + c) / s


episode = 0

# Main training loop
for episode in range(start + 1, flags.num_episodes + 1):
    agent.reset()
    env_renderer.reset()
    obs, info = env.reset(True, True)
    score, steps_taken, collision = 0, 0, False

    agent_obs = [normalize_observation(obs[a], flags.tree_depth, zero_center=True)
                 for a in obs.keys()]
    agent_obs_buffer = copy.deepcopy(agent_obs)
    agent_count = len(agent_obs)
    agent_action_buffer = [2] * agent_count

    # Run an episode
    max_steps = 8 * (env.width + env.height)
    for step in range(max_steps):
        update_values = [False] * agent_count
        action_dict = {}

        if any(info['action_required']):
            ret_action = agent.act(agent_obs, eps=eps)
        else:
            ret_action = update_values
        for idx, act in enumerate(ret_action):
            if info['action_required'][idx]:
                action_dict[idx] = act
                # action_dict[a] = np.random.randint(5)
                update_values[idx] = True
                steps_taken += 1
            else:
                action_dict[idx] = 0

        # Environment step
        obs, rewards, done, info = env.step(action_dict)
        score += sum(rewards.values()) / agent_count

        # Check for collisions and episode completion
        if step == max_steps - 1:
            done['__all__'] = True
        if any(is_collision(a) for a in obs):
            collision = True
            # done['__all__'] = True

        # Update replay buffer and train agent
        if flags.train and (any(update_values) or any(done) or done['__all__']):
            agent.step(agent_obs_buffer,
                       agent_action_buffer,
                       agent_obs,
                       done,
                       done['__all__'],
                       [is_collision(a) for a in range(agent_count)],
                       flags.step_reward)
            agent_obs_buffer = [o.clone() for o in agent_obs]
            for key, value in action_dict.items():
                agent_action_buffer[key] = value

        for a in range(agent_count):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], flags.tree_depth, zero_center=True)

    # Render
    # if flags.render_interval and episode % flags.render_interval == 0:
    # if collision and all(agent.position for agent in env.agents):
    #     render()
    #     print("Collisions detected by agent(s)", ', '.join(str(a) for a in obs if is_collision(a)))
    #     break
        if done['__all__']:
            break

    # Epsilon decay
    if flags.train:
        eps = max(0.01, flags.epsilon_decay * eps)

    # Save some training statistics in their respective deques
    tasks_finished = sum(done[i] for i in range(agent_count))
    current_done, mean_done = get_means(current_done, mean_done, tasks_finished / max(1, agent_count), episode)
    current_collisions, mean_collisions = get_means(current_collisions, mean_collisions, int(collision), episode)
    current_score, mean_score = get_means(current_score, mean_score, score / max_steps, episode)
    current_steps, mean_steps = get_means(current_steps, mean_steps, steps_taken, episode)

    print(f'\rEpisode {episode:<5}'
          f' | Score: {current_score:.4f}, {mean_score:.4f}'
          f' | Steps: {current_steps:6.1f}, {mean_steps:6.1f}'
          f' | Collisions: {100 * current_collisions:5.2f}%, {100 * mean_collisions:5.2f}%'
          f' | Finished: {100 * current_done:6.2f}%, {100 * mean_done:6.2f}%'
          f' | Epsilon: {eps:.2f}'
          f' | Episode/s: {episode / (time.time() - start_time):.2f}s', end='')

    if episode % flags.report_interval == 0:
        print("")
        if flags.train:
            agent.save(project_root / 'checkpoints', episode, eps)
        # Add stats to the tensorboard summary
