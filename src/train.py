import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from tensorboardX import SummaryWriter

from flatland.envs.rail_env import RailEnv
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from dqn.agent import Agent as DQN_Agent
from ppo.agent import Agent as PPO_Agent
from tree_observation import TreeObservation
from observation_utils import normalize_observation, is_collision
from railway_utils import load_precomputed_railways, create_random_railways


project_root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description="Train an agent in the flatland environment")
boolean = lambda x: str(x).lower() == 'true'

# Task parameters
parser.add_argument("--train", type=boolean, default=True, help="Whether to train the model or just evaluate it")
parser.add_argument("--load-model", type=boolean, default=False, help="Whether to load the model from the last checkpoint")
parser.add_argument("--load-railways", type=boolean, default=True, help="Whether to load in pre-generated railway networks")
parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")
parser.add_argument("--render-interval", type=int, default=0, help="Iterations between renders")

# Environment parameters
parser.add_argument("--grid-width", type=int, default=50, help="Number of columns in the environment grid")
parser.add_argument("--grid-height", type=int, default=50, help="Number of rows in the environment grid")
parser.add_argument("--num-agents", type=int, default=2, help="Number of agents in each episode")
parser.add_argument("--tree-depth", type=int, default=1, help="Depth of the observation tree")

# Training parameters
parser.add_argument("--agent-type", default="ppo", choices=["dqn", "ppo"], help="Which type of RL agent to use")
parser.add_argument("--num-episodes", type=int, default=10000, help="Number of episodes to train for")
parser.add_argument("--epsilon-decay", type=float, default=0.997, help="Decay factor for epsilon-greedy exploration")

flags = parser.parse_args()


# Seeded RNG so we can replicate our results
np.random.seed(1)

# Create a tensorboard SummaryWriter
summary = SummaryWriter(f'tensorboard/dqn/agents: {flags.num_agents}, tree_depth: {flags.tree_depth}')

# We need to either load in some pre-generated railways from disk, or else create a random railway generator.
if flags.load_railways:
      rail_generator, schedule_generator = load_precomputed_railways(project_root, flags)
else: rail_generator, schedule_generator = create_random_railways(project_root)

# Create the Flatland environment
env = RailEnv(width=flags.grid_width, height=flags.grid_height, number_of_agents=flags.num_agents,
              remove_agents_at_target=True,
              rail_generator=rail_generator,
              schedule_generator=schedule_generator,
              malfunction_generator_and_process_data=malfunction_from_params(MalfunctionParameters(1 / 8000, 15, 50)),
              obs_builder_object=TreeObservation(max_depth=flags.tree_depth))

# After training we want to render the results so we also load a renderer
env_renderer = RenderTool(env, gl="PILSVG", agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX)

# Calculate the state size based on the number of nodes in the tree observation
num_features_per_node = env.obs_builder.observation_dim
num_nodes = sum(np.power(4, i) for i in range(flags.tree_depth + 1))
state_size = num_nodes * num_features_per_node
action_size = 5

# Add some variables to keep track of the progress
scores_window, steps_window, collisions_window, done_window = [deque(maxlen=200) for _ in range(4)]
agent_obs = [None] * flags.num_agents
agent_obs_buffer = [None] * flags.num_agents
agent_action_buffer = [2] * flags.num_agents
max_steps = 8 * (flags.grid_width + flags.grid_height)
start_time = time.time()

# Load an RL agent and initialize it from checkpoint if necessary
if flags.agent_type == "dqn":
    agent = DQN_Agent(state_size, action_size, flags.num_agents)
elif flags.agent_type == "ppo":
    agent = PPO_Agent(state_size, action_size, flags.num_agents)

if flags.load_model:
      start, eps = agent.load(project_root / 'checkpoints', 0, 1.0)
else: start, eps = 0, 1.0

if not flags.train:
    eps = 0.0

# We don't want to retrain on old railway networks when we restart from a checkpoint, so we just loop
# through the generators to get all the old networks out of the way
if start > 0: print(f"Skipping {start} railways")
for _ in range(0, start):
    rail_generator()
    schedule_generator()

# Helper function to generate a report
def get_report(show_time=False):
    training = 'Training' if flags.train else 'Evaluating'
    return '  |  '.join(filter(None, [
        f'\r{training} {flags.num_agents} Agents on {flags.grid_width} x {flags.grid_height} Map',
        f'Episode {episode:<5}',
        f'Average Score: {np.mean(scores_window):.3f}',
        f'Average Steps Taken: {np.mean(steps_window):<6.1f}',
        f'Collisions: {100 * np.mean(collisions_window):>5.2f}%',
        f'Finished: {100 * np.mean(done_window):>6.2f}%',
        f'Epsilon: {eps:.2f}' if flags.agent_type == "dqn" else None,
        f'Time taken: {time.time() - start_time:.2f}s' if show_time else None])) + '  '



# Main training loop
for episode in range(start + 1, flags.num_episodes + 1):
    agent.reset()
    env_renderer.reset()
    obs, info = env.reset(True, True)
    score, steps_taken, collision = 0, 0, False

    # Build initial observations for each agent
    for a in range(flags.num_agents):
        agent_obs[a] = normalize_observation(obs[a], flags.tree_depth, zero_center=flags.agent_type == 'dqn')
        agent_obs_buffer[a] = agent_obs[a].copy()

    # Run an episode
    for step in range(max_steps):
        update_values = [False] * flags.num_agents
        action_dict = {}

        for a in range(flags.num_agents):
            if info['action_required'][a]:
                  action_dict[a] = agent.act(agent_obs[a], eps=eps)
                  # action_dict[a] = np.random.randint(5)
                  update_values[a] = True
                  steps_taken += 1
            else: action_dict[a] = 0

        # Environment step
        obs, rewards, done, info = env.step(action_dict)
        score += sum(rewards.values()) / flags.num_agents

        # Check for collisions and episode completion
        if step == max_steps - 1:
            done['__all__'] = True
        if any(is_collision(obs[a]) for a in obs):
            collision = True

        # Update replay buffer and train agent
        for a in range(flags.num_agents):
            if flags.train and (update_values[a] or done[a] or done['__all__']):
                agent.step(a, agent_obs_buffer[a], agent_action_buffer[a], agent_obs[a], done[a], done['__all__'], is_collision(obs[a]))
                agent_obs_buffer[a] = agent_obs[a].copy()
                agent_action_buffer[a] = action_dict[a]

            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], flags.tree_depth, zero_center=flags.agent_type == 'dqn')

        # Render
        if flags.render_interval and episode % flags.render_interval == 0:
            env_renderer.render_env(show_observations=True)
            cv2.imshow('Render', cv2.cvtColor(env_renderer.get_image(), cv2.COLOR_BGR2RGB))
            cv2.waitKey(30)
        if done['__all__']: break

    # Epsilon decay
    if flags.train: eps = max(0.01, flags.epsilon_decay * eps)

    # Save some training statistics in their respective deques
    tasks_finished = sum(done[i] for i in range(flags.num_agents))
    done_window.append(tasks_finished / max(1, flags.num_agents))
    collisions_window.append(1. if collision else 0.)
    scores_window.append(score / max_steps)
    steps_window.append(steps_taken)

    # Generate training reports, saving our progress every so often
    print(get_report(), end=" ")
    if episode % flags.report_interval == 0:
        print(get_report(show_time=True))
        start_time = time.time()
        if flags.train: agent.save(project_root / 'checkpoints', episode, eps)

    # Add stats to the tensorboard summary
    summary.add_scalar('performance/avg_score', np.mean(scores_window), episode)
    summary.add_scalar('performance/avg_steps', np.mean(steps_window), episode)
    summary.add_scalar('performance/completions', np.mean(done_window), episode)
    summary.add_scalar('performance/collisions', np.mean(collisions_window), episode)
