import collections
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent

tree_explored_actions_char = ['L', 'F', 'R', 'B']

Node = collections.namedtuple('Node', ('dist_own_target_encountered',
                                       'dist_other_target_encountered',
                                       'dist_other_agent_encountered',
                                       'dist_potential_conflict',
                                       'dist_unusable_switch',
                                       'dist_to_next_branch',
                                       'dist_min_to_target',
                                       'num_agents_same_direction',
                                       'num_agents_opposite_direction',
                                       'num_agents_malfunctioning',
                                       'speed_min_fractional',
                                       'num_agents_ready_to_depart',
                                       'children'))


def reverse_dir(self, direction):
    return int((direction + 2) % 4)


class TreeObservation(ObservationBuilder):
    def __init__(self, max_depth, predictor = None):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = predictor
        self.observation_dim = 11
        self.target_positions = {}
        self.agent_positions = {}
        self.agent_directions = {}

    def reset(self):
        self.target_positions = { tuple(agent.target): 1 for agent in self.env.agents }

    def get_many(self, handles = None):
        if handles is None:
            handles = []

        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is not None:
                            pos_list.append(self.predictions[a][t][1:3])
                            dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({ t: coordinate_to_position(self.env.width, pos_list) })
                    self.predicted_dir.update({ t: dir_list })
                self.max_prediction_depth = len(self.predicted_pos)

        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.agent_positions = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.agent_positions = {}
        self.agent_directions = {}
        self.agent_speeds = {}
        self.agent_malfunctions = {}
        self.agents_ready_to_depart = {}

        for agent in self.env.agents:
            if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
                self.agent_positions[agent.position] = 1
                self.agent_directions[agent.position] = agent.direction
                self.agent_speeds[agent.position] = agent.speed_data['speed']
                self.agent_malfunctions[agent.position] = agent.malfunction_data['malfunction']

            if agent.status in [RailAgentStatus.READY_TO_DEPART] and agent.initial_position:
                self.agents_ready_to_depart[agent.initial_position] = \
                    self.agents_ready_to_depart.get(agent.initial_position, 0) + 1

        return super().get_many(handles)

    # Called by ObservationBuilder.get_many to compute the observation for a single agent
    def get(self, handle):
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
              agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
              agent_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
              agent_position = agent.target
        else: return None

        possible_transitions = self.env.rail.get_transitions(*agent_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # The root node contains information about the agent itself
        dist_min_to_target = self.env.distance_map.get()[(handle, *agent_position, agent.direction)]
        agent_malfunctioning, agent_speed = agent.malfunction_data['malfunction'], agent.speed_data['speed']
        root_node_observation = Node(0, 0, 0, 0, 0, 0, dist_min_to_target, 0, 0, agent_malfunctioning, agent_speed, 0, {})

        visited = set()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = np.argmax(possible_transitions) if num_transitions == 1 else agent.direction

        for i, branch_direction in enumerate((orientation + i) % 4 for i in range(-1, 3)):
            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_position, branch_direction)
                branch_observation, branch_visited = self.explore_branch(handle, new_cell, branch_direction, 1, 1)
                root_node_observation.children[tree_explored_actions_char[i]] = branch_observation
                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.children[tree_explored_actions_char[i]] = -np.inf

        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

    def explore_branch(self, handle, position, direction, total_distance, depth):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """

        # Base condition
        if depth > self.max_depth:
            return [], []

        # Continue along this direction until the next switch, or until no more transitions are possible (i.e., dead-ends).
        # We treat dead-ends as nodes, instead of going back, to avoid loops.
        last_is_target = False
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False

        visited = set()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0

        # Keep exploring down this path until we reach the next switch / target / dead end
        while True:
            if position in self.agent_positions:
                other_agent_encountered = min(other_agent_encountered, total_distance)
                malfunctioning_agent = max(malfunctioning_agent, self.agent_malfunctions[position])

                if position in self.agents_ready_to_depart:
                      other_agent_ready_to_depart_encountered += 1

                if self.agent_directions[position] == direction:
                      other_agent_same_direction += 1
                      min_fractional_speed = min(min_fractional_speed, self.agent_speeds[position])
                else: other_agent_opposite_direction += 1

            # Check number of possible transitions for agent and total number of transitions in cell
            possible_transitions = self.env.rail.get_transitions(*position, direction)
            num_transitions = np.count_nonzero(possible_transitions)

            transition_bits = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bits.count("1")
            crossing_found = int(transition_bits, 2) == int('1000010000100001', 2)

            # Register possible future conflict
            predicted_time = int(total_distance * time_per_cell)
            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if total_distance < self.max_prediction_depth:
                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                    # Look for conflicting paths at distance total_distance
                    if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[predicted_time][ca] \
                                and possible_transitions[reverse_dir(self.predicted_dir[predicted_time][ca])] == 1 \
                                and total_distance < potential_conflict:
                                potential_conflict = total_distance
                            if self.env.agents[ca].status == RailAgentStatus.DONE and total_distance < potential_conflict:
                                potential_conflict = total_distance

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] \
                                and possible_transitions[reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                and total_distance < potential_conflict:
                                potential_conflict = total_distance
                            if self.env.agents[ca].status == RailAgentStatus.DONE and total_distance < potential_conflict:
                                potential_conflict = total_distance

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] \
                                and possible_transitions[reverse_dir(self.predicted_dir[post_step][ca])] == 1 \
                                and total_distance < potential_conflict:
                                potential_conflict = total_distance
                            if self.env.agents[ca].status == RailAgentStatus.DONE and total_distance < potential_conflict:
                                potential_conflict = total_distance

            if position in self.target_positions and position != agent.target:
                other_target_encountered = min(other_target_encountered, total_distance)
            if position == agent.target:
                own_target_encountered = min(own_target_encountered, total_distance)

            if (*position, direction) in visited:
                last_is_terminal = True
                break

            visited.add((*position, direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Detect switches that can only be used by other agents.
            if not crossing_found and num_transitions < 2 < total_transitions and total_distance < unusable_switch:
                unusable_switch = total_distance

            # Check if we've found a tree node (switch / target / dead end)
            if num_transitions > 1:
                last_is_switch = True
                break

            elif num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                if total_transitions == 1:
                    last_is_dead_end = True
                    break
                else:
                    direction = np.argmax(possible_transitions)
                    position = get_new_position(position, direction)
                    num_steps += 1
                    total_distance += 1

            # These shouldn't exist, dead ends should let you exit backwards
            elif num_transitions == 0:
                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            dist_to_next_branch = total_distance
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
        else:
            dist_to_next_branch = total_distance
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]

        node = Node(dist_own_target_encountered=own_target_encountered,
                    dist_other_target_encountered=other_target_encountered,
                    dist_other_agent_encountered=other_agent_encountered,
                    dist_potential_conflict=potential_conflict,
                    dist_unusable_switch=unusable_switch,
                    dist_to_next_branch=dist_to_next_branch,
                    dist_min_to_target=dist_min_to_target,
                    num_agents_same_direction=other_agent_same_direction,
                    num_agents_opposite_direction=other_agent_opposite_direction,
                    num_agents_malfunctioning=malfunctioning_agent,
                    speed_min_fractional=min_fractional_speed,
                    num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                    children={})


        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if last_is_dead_end and self.env.rail.get_transition((*position, direction), (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes it back
                new_cell = get_new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = \
                    self.explore_branch(handle, new_cell, (branch_direction + 2) % 4, total_distance + 1, depth + 1)
                node.children[tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited

            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation, branch_visited = \
                    self.explore_branch(handle, new_cell, branch_direction, total_distance + 1, depth + 1)
                node.children[tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited):
                    visited |= branch_visited

            else:
                # no exploring possible, add just cells with infinity
                node.children[tree_explored_actions_char[i]] = -np.inf

        if depth == self.max_depth:
            node.children.clear()

        return node, visited


    def util_print_obs_subtree(self, tree: Node):
        self.print_node_features(tree, "root", "")
        for direction in tree_explored_actions_char:
            self.print_subtree(tree.children[direction], direction, "\t")

    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.children:
            return

        for direction in tree_explored_actions_char:
            self.print_subtree(node.children[direction], direction, indent + "\t")

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)
