import cv2
import time
import torch
import pickle
import numpy as np
from pathlib import Path
from collections import deque, namedtuple

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator, complex_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.utils.rendertools import RenderTool

from dueling_double_dqn import Agent
from tree_observation import TreeObservation
from observation_utils import normalize_observation

from tensorboardX import SummaryWriter 

# Parameters for the environment
n_trials = 10000
n_agents = 1
x_dim = 35
y_dim = 35
tree_depth = 2
eps_decay = 0.999
eps_end = 0.005

report_interval = 100
render_interval = 1000
load_from_checkpoint = False
train = True


project_root = Path(__file__).parent.parent
StochasticData = namedtuple('StochasticData', ('malfunction_rate', 'min_duration', 'max_duration'))

# Load in the precomputed railway networks. If you want to generate railways on the fly, comment these lines out.
with open(project_root / f'railroads/rail_networks_{n_agents}x{x_dim}x{y_dim}.pkl', 'rb') as file:
    data = pickle.load(file)
    rail_networks = iter(data)
    print(f"Loading {len(data)} railways...")
with open(project_root / f'railroads/schedules_{n_agents}x{x_dim}x{y_dim}.pkl', 'rb') as file:
    schedules = iter(pickle.load(file))

rail_generator = lambda *args: next(rail_networks)
schedule_generator = lambda *args: next(schedules)


def render(env_renderer):
    env_renderer.render_env()
    cv2.imshow('Render', env_renderer.get_image())
    cv2.waitKey(100)

def main():
    np.random.seed(1)

    writer = SummaryWriter('tensorboard_logs/dqn/agents: {}, tree_depth: {}'.format(n_agents,tree_depth))

    env = RailEnv(width=x_dim, height=y_dim, number_of_agents=n_agents,
                  rail_generator=rail_generator,
                  schedule_generator=schedule_generator,
                  malfunction_generator_and_process_data=malfunction_from_params(StochasticData(1 / 8000, 15, 50)),
                  obs_builder_object=TreeObservation(max_depth=tree_depth))

    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG")

    # Calculate the state size based on the number of nodes in the tree observation
    num_features_per_node = env.obs_builder.observation_dim
    num_nodes = sum(np.power(4, i) for i in range(tree_depth + 1))
    state_size = num_features_per_node * num_nodes
    action_size = 5

    # Now we load a double dueling DQN agent and initialize it from the checkpoint
    agent = Agent(state_size, action_size)
    if load_from_checkpoint:
          start, eps = agent.load(project_root / 'checkpoints', 0, 1.0)
    else: start, eps = 0, 1.0

    # And some variables to keep track of the progress
    action_dict, final_action_dict = {}, {}
    scores_window, done_window = deque(maxlen=500), deque(maxlen=500)
    action_prob = [0] * action_size
    agent_obs = [None] * n_agents
    agent_obs_buffer = [None] * n_agents
    agent_action_buffer = [2] * n_agents

    max_steps = int(3 * (x_dim + y_dim))
    update_values = False
    start_time = time.time()

    # We don't want to retrain on old railway networks when we restart from a checkpoint, so we just loop
    # through the generators to get all the old networks out of the way
    for _ in range(0, start):
        rail_generator()
        schedule_generator()

    # Start the training loop
    for episode in range(start + 1, n_trials + 1):
        env_renderer.reset()
        obs, info = env.reset(True, True)
        score = 0

        # Build agent specific observations
        for a in range(n_agents):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
                agent_obs_buffer[a] = agent_obs[a].copy()

        # Run episode
        for step in range(max_steps):
            for a in range(n_agents):
                if info['action_required'][a]:
                    # If an action is required, we want to store the obs a that step as well as the action
                    update_values = True
                    action = agent.act(agent_obs[a], eps=eps)
                    # action = np.random.randint(4)
                    action_dict[a] = action
                    action_prob[action] += 1
                else:
                    update_values = False
                    action_dict[a] = 0

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)

            # Update replay buffer and train agent
            for a in range(n_agents):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values or done[a]:
                    agent.step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a], agent_obs[a], done[a], train)
                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]
                if next_obs[a]:
                    agent_obs[a] = normalize_observation(next_obs[a], tree_depth, observation_radius=10)

                score += all_rewards[a] / n_agents

            # Render
            if episode % render_interval == 0: render(env_renderer)
            if done['__all__']: break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Collection information about training
        tasks_finished = sum(done[i] for i in range(n_agents))
        done_window.append(tasks_finished / max(1, n_agents))
        scores_window.append(score / max_steps)  # save most recent score

        action_probs = ', '.join(f'{x:.3f}' for x in action_prob / np.sum(action_prob))
        print(f'\rTraining {n_agents} Agents on ({x_dim},{y_dim}) \t ' +
              f'Episode {episode} \t ' +
              f'Average Score: {np.mean(scores_window):.3f} \t ' +
              f'Dones: {100 * np.mean(done_window):.2f}% \t ' +
              f'Epsilon: {eps:.2f} \t ' +
              f'Action Probabilities: {action_probs}', end=" ")

        writer.add_scalar('performance/avg_score',np.mean(scores_window),episode)   
        writer.add_scalar('performance/completions',np.mean(done_window),episode)   
        [writer.add_scalar(f'action_probabilites/action_{i}',action_prob[i]/np.sum(action_prob),episode) for i in range(len(action_prob))]


        if episode % report_interval == 0:
            print(f'\rTraining {n_agents} Agents on ({x_dim},{y_dim}) \t ' +
                  f'Episode {episode} \t ' +
                  f'Average Score: {np.mean(scores_window):.3f} \t ' +
                  f'Dones: {100 * np.mean(done_window):.2f}% \t ' +
                  f'Epsilon: {eps:.2f} \t ' +
                  f'Action Probabilities: {action_probs} \t ' +
                  f'Time taken: {time.time() - start_time:.2f}s')

            if train: agent.save(project_root / 'checkpoints', episode, eps)
            start_time = time.time()
            action_prob = [1] * action_size


if __name__ == '__main__':
    main()
