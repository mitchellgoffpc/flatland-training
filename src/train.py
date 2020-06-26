import cv2
import time
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import deque, namedtuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters
# from flatland.envs.observations import TreeObsForRailEnv as TreeObservation
from flatland.utils.rendertools import RenderTool

from dqn.agent import Agent
# from ppo.agent import Agent
from tree_observation import TreeObservation
from observation_utils import normalize_observation


parser = argparse.ArgumentParser(description="Train an agent in the flatland environment")

# Task parameters
parser.add_argument('--eval', default=True, dest='train', action='store_false', help="Evaluate the model")
parser.add_argument("--load-from-checkpoint", default=False, action='store_true', help="Whether to load the model from the last checkpoint")
parser.add_argument("--report-interval", type=int, default=100, help="Iterations between reports")
parser.add_argument("--render-interval", type=int, default=0, help="Iterations between renders")

# Environment parameters
parser.add_argument("--grid-width", type=int, default=50, help="Number of columns in the environment grid")
parser.add_argument("--grid-height", type=int, default=50, help="Number of rows in the environment grid")
parser.add_argument("--num-agents", type=int, default=2, help="Number of agents in each episode")
parser.add_argument("--tree-depth", type=int, default=2, help="Depth of the observation tree")

# Training parameters
parser.add_argument("--num-episodes", type=int, default=10000, help="Number of episodes to train for")
parser.add_argument("--epsilon-decay", type=float, default=0.997, help="Decay factor for epsilon-greedy exploration")

flags = parser.parse_args()
project_root = Path(__file__).resolve().parent.parent


# Load in the precomputed railway networks. If you want to generate railways on the fly, comment these lines out.
with open(project_root / f'railroads/rail_networks_{flags.num_agents}x{flags.grid_width}x{flags.grid_height}.pkl', 'rb') as file:
    data = pickle.load(file)
    rail_networks = iter(data)
    print(f"Loading {len(data)} railways...")
with open(project_root / f'railroads/schedules_{flags.num_agents}x{flags.grid_width}x{flags.grid_height}.pkl', 'rb') as file:
    schedules = iter(pickle.load(file))

rail_generator = lambda *args: next(rail_networks)
schedule_generator = lambda *args: next(schedules)

# Generate railways on the fly
# speed_ration_map = {
#     1 / 1:  1.0,   # Fast passenger train
#     1 / 2.: 0.0,   # Fast freight train
#     1 / 3.: 0.0,   # Slow commuter train
#     1 / 4.: 0.0 }  # Slow freight train
#
# rail_generator = sparse_rail_generator(grid_mode=False, max_num_cities=3, max_rails_between_cities=2, max_rails_in_city=3)
# schedule_generator = sparse_schedule_generator(speed_ration_map)


# Helper function to render the environment
def render(env_renderer):
    env_renderer.render_env(show_observations=True)
    cv2.imshow('Render', cv2.cvtColor(env_renderer.get_image(), cv2.COLOR_BGR2RGB))
    cv2.waitKey(30)


# Main training loop
def main():
    np.random.seed(1)

    env = RailEnv(width=flags.grid_width, height=flags.grid_height, number_of_agents=flags.num_agents,
                  remove_agents_at_target=True,
                  rail_generator=rail_generator,
                  schedule_generator=schedule_generator,
                  malfunction_generator_and_process_data=malfunction_from_params(MalfunctionParameters(1 / 8000, 15, 50)),
                  obs_builder_object=TreeObservation(max_depth=flags.tree_depth))

    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG")

    # Calculate the state size based on the number of nodes in the tree observation
    num_features_per_node = env.obs_builder.observation_dim
    num_nodes = sum(np.power(4, i) for i in range(flags.tree_depth + 1))
    state_size = num_nodes * num_features_per_node
    action_size = 5

    # Now we load a double dueling DQN agent and initialize it from the checkpoint
    agent = Agent(state_size, action_size, flags.num_agents)
    if flags.load_from_checkpoint:
          start, eps = agent.load(project_root / 'checkpoints', 0, 1.0)
    else: start, eps = 0, 1.0

    if not flags.train: eps = 0.0

    # And some variables to keep track of the progress
    scores_window, steps_window, collisions_window, done_window = [deque(maxlen=200) for _ in range(4)]
    agent_obs = [None] * flags.num_agents
    agent_obs_buffer = [None] * flags.num_agents
    agent_action_buffer = [2] * flags.num_agents
    max_steps = 8 * (flags.grid_width + flags.grid_height)
    start_time = time.time()

    # We don't want to retrain on old railway networks when we restart from a checkpoint, so we just loop
    # through the generators to get all the old networks out of the way
    if start > 0: print(f"Skipping {start} railways")
    for _ in range(0, start):
        rail_generator()
        schedule_generator()

    # Start the training loop
    for episode in range(start + 1, flags.num_episodes + 1):
        agent.reset()
        env_renderer.reset()
        obs, info = env.reset(True, True)
        score, steps_taken, collision = 0, 0, False

        # Build agent specific observations
        for a in range(flags.num_agents):
            agent_obs[a] = normalize_observation(obs[a], flags.tree_depth)
            agent_obs_buffer[a] = agent_obs[a].copy()

        # Run episode
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
            obs, all_rewards, done, info = env.step(action_dict)

            if step == max_steps - 1:
                done['__all__'] = True

            # Calculate rewards
            rewards = [0] * flags.num_agents
            for a in range(flags.num_agents):
                if done[a] and not agent.finished[a]:
                      rewards[a] = 1
                elif not done[a] \
                     and isinstance(obs[a].childs['L'], float) \
                     and isinstance(obs[a].childs['R'], float) \
                     and obs[a].childs['F'].dist_other_agent_encountered <= 1 \
                     and obs[a].childs['F'].dist_other_agent_encountered < obs[a].childs['F'].dist_unusable_switch \
                     and obs[a].childs['F'].num_agents_opposite_direction > 0:
                      # done['__all__'] = True
                      # rewards[a] = -.5 - episode / 1500
                      # rewards[a] = -1
                      collision = True
                      rewards[a] = -1
                else: rewards[a] = -.02

            # Update replay buffer and train agent
            for a in range(flags.num_agents):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                finished = done['__all__'] or done[a]
                if update_values[a] or (finished and not agent.finished[a]):
                    agent.step(a, agent_obs_buffer[a], agent_action_buffer[a], rewards[a], agent_obs[a], finished, flags.train)
                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]
                if obs[a]:
                    agent_obs[a] = normalize_observation(obs[a], flags.tree_depth)

                score += all_rewards[a] / flags.num_agents

            # Render
            if flags.render_interval and episode % flags.render_interval == 0:
            # if collision:
                render(env_renderer)
            if done['__all__']: break

        # Epsilon decay
        if flags.train:
            eps = max(0.01, flags.epsilon_decay * eps)

        # Save some training statistics in their respective deques
        tasks_finished = sum(done[i] for i in range(flags.num_agents))
        done_window.append(tasks_finished / max(1, flags.num_agents))
        scores_window.append(score / max_steps)
        steps_window.append(steps_taken)
        collisions_window.append(1. if collision else 0.)

        print(f'\rTraining {flags.num_agents} Agents on ({flags.grid_width},{flags.grid_height}) \t ' +
              f'Episode {episode} \t ' +
              f'Average Score: {np.mean(scores_window):.3f} \t ' +
              f'Average Steps Taken: {np.mean(steps_window):.1f} \t ' +
              f'Collisions: {100 * np.mean(collisions_window):.2f}% \t ' +
              f'Finished: {100 * np.mean(done_window):.2f}% \t ' +
              f'Epsilon: {eps:.2f}', end=" ")

        if episode % flags.report_interval == 0:
            print(f'\rTraining {flags.num_agents} Agents on ({flags.grid_width},{flags.grid_height}) \t ' +
                  f'Episode {episode} \t ' +
                  f'Average Score: {np.mean(scores_window):.3f} \t ' +
                  f'Average Steps Taken: {np.mean(steps_window):.1f} \t ' +
                  f'Collisions: {100 * np.mean(collisions_window):.2f}% \t ' +
                  f'Finished: {100 * np.mean(done_window):.2f}% \t ' +
                  f'Epsilon: {eps:.2f} \t ' +
                  f'Time taken: {time.time() - start_time:.2f}s')

            if flags.train: agent.save(project_root / 'checkpoints', episode, eps)
            start_time = time.time()


if __name__ == '__main__':
    main()
