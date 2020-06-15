import pprint
import numpy as np
from deepdiff import DeepDiff
from collections import defaultdict

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv


ACTIONS = ['L', 'F', 'R', 'B']
Node = TreeObsForRailEnv.Node
printer = pprint.PrettyPrinter(indent=4)

def first(list):
    return next(iter(list))

def get_action(orientation, direction):
    return ACTIONS[(direction - orientation + 1 + 4) % 4]


class RailNode:
    def __init__(self, position, edge_directions, is_target):
        self.edges = {}
        self.position = position
        self.edge_directions = edge_directions
        self.is_target = is_target

    def __repr__(self):
        return f'RailNode({self.position}, {len(self.edges)})'



class TreeObservation(ObservationBuilder):
    def __init__(self, max_depth, predictor = None):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = predictor
        self.observation_dim = 11
        self.original_obs = TreeObsForRailEnv(max_depth=max_depth)


    # Create a graph representation of the current rail network

    def reset(self):
        self.original_obs.reset()
        self.target_positions = { agent.target: 1 for agent in self.env.agents }
        self.edge_positions = defaultdict(list) # (cell.position, direction) -> [(start, end, direction, distance)]
        self.edge_paths = defaultdict(list)     # (node.position, direction) -> [(cell.position, direction)]

        # First, we find a node by starting at one of the agents and following the rails until we reach a junction
        agent = first(self.env.agents)
        position = agent.initial_position
        direction = agent.direction
        while not self.is_junction(position) and not self.is_target(position):
            direction = first(self.get_possible_transitions(position, direction))
            position = get_new_position(position, direction)

        # Now we create a graph representation of the rail network, starting from this node
        transitions = self.get_all_transitions(position)
        root_nodes = { t: RailNode(position, t, self.is_target(position)) for t in transitions if t }
        self.graph = { (*position, d): root_nodes[t] for d, t in enumerate(transitions) if t }

        for transitions, node in root_nodes.items():
            for direction in transitions:
                self.explore_branch(node, get_new_position(position, direction), direction)


    def explore_branch(self, node, position, direction):
        original_direction = direction
        edge_positions = {}
        distance = 1

        # Explore until we find a junction
        while not self.is_junction(position) and not self.is_target(position):
            edge_positions[(*position, direction)] = distance
            direction = first(self.get_possible_transitions(position, direction))
            position = get_new_position(position, direction)
            distance += 1

        # Create any nodes that aren't in the graph yet
        transitions = self.get_all_transitions(position)
        nodes = { t: RailNode(position, t, self.is_target(position))
                  for d, t in enumerate(transitions)
                  if t and (*position, d) not in self.graph }

        for d, t in enumerate(transitions):
            if t in nodes:
                self.graph[(*position, d)] = nodes[t]

        # Connect the previous node to the next one, and update self.edge_positions
        next_node = self.graph[(*position, direction)]
        node.edges[original_direction] = (next_node, distance)
        for key, distance in edge_positions.items():
            self.edge_positions[key].append((node, next_node, original_direction, distance))
            self.edge_paths[node.position, original_direction].append(key)

        # Call ourselves recursively since we're exploring depth-first
        for transitions, node in nodes.items():
            for direction in transitions:
                self.explore_branch(node, get_new_position(position, direction), direction)


    # Create a tree observation for each agent, based on the graph we created earlier

    def get_many(self, handles = []):
        self.nodes_with_agents_going, self.nodes_with_agents_coming = {}, {}
        self.edges_with_agents_going, self.edges_with_agents_coming = defaultdict(dict), defaultdict(dict)
        self.edges_with_malfunctions = defaultdict(dict)

        # Create some lookup tables that we can use later to figure out how far away the agents are from each other.
        for agent in self.env.agents:
            if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
                for direction in range(4):
                    if (*agent.position, direction) in self.graph:
                        node_dict = self.nodes_with_agents_going if direction == agent.direction else self.nodes_with_agents_coming
                        node_dict[(*agent.position, direction)] = agent.speed_data['speed']

                    edge_dict = self.edges_with_agents_going if direction == agent.direction else self.edges_with_agents_coming
                    for start, _, start_direction, distance in self.edge_positions[(*agent.position, direction)]:
                        edge_dict[(*start.position, start_direction)][agent.handle] = (distance, agent.speed_data['speed'])

                if agent.malfunction_data['malfunction']:
                    for start, _, direction, distance in self.edge_positions[(*agent.position, agent.direction)]:
                        self.edges_with_malfunctions[(*start.position, direction)][agent.handle] = distance

        my_tree = super().get_many(handles)
        original_tree = self.original_obs.get_many(handles)
        diff = DeepDiff(original_tree, my_tree)
        if len(diff):
            print([agent.position for agent in self.env.agents])
            printer.pprint(diff)
            raise Exception("diff complete")

        return my_tree

    # Compute the observation for a single agent
    def get(self, handle):
        agent = self.env.agents[handle]
        visited_cells = set()

        if agent.status == RailAgentStatus.READY_TO_DEPART:
              agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
              agent_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
              agent_position = agent.target
        else: return None

        # The root node contains information about the agent itself
        children = { x: -np.inf for x in ACTIONS }
        dist_min_to_target = self.env.distance_map.get()[(handle, *agent_position, agent.direction)]
        agent_malfunctioning, agent_speed = agent.malfunction_data['malfunction'], agent.speed_data['speed']
        root_tree_node = Node(0, 0, 0, 0, 0, 0, dist_min_to_target, 0, 0, agent_malfunctioning, agent_speed, 0, children)

        # Then we build out the tree by exploring from this node
        key = (*agent_position, agent.direction)
        if key in self.graph: # If we're sitting on a junction, branch out immediately
            node = self.graph[key]
            if len(node.edges) > 1: # Major node
                for direction in self.graph[key].edges.keys():
                    root_tree_node.childs[get_action(agent.direction, direction)] = \
                        self.get_tree_branch(agent, node, direction, visited_cells, 0, 1)
            else: # Minor node
                direction = first(self.get_possible_transitions(node.position, agent.direction))
                root_tree_node.childs['F'] = self.get_tree_branch(agent, node, direction, visited_cells, 0, 1)

        else: # Just create a single child in the forward direction
            prev_node, next_node, direction, distance = first(self.edge_positions[key])
            path = self.edge_paths[prev_node.position, direction]
            root_tree_node.childs['F'] = self.get_tree_branch(agent, prev_node, direction, visited_cells, -distance, 1)

        self.env.dev_obs_dict[handle] = visited_cells

        return root_tree_node

    # Get the next tree node, starting from `node`, facing `orientation`, and moving in `direction`.
    def get_tree_branch(self, agent, node, direction, visited_cells, total_distance, depth):
        visited_cells.add((*node.position, 0))
        next_node, distance = node.edges[direction]
        original_position = node.position

        targets, agents, minor_nodes = [], [], []
        edge_length, num_malfunctions = 0, 0
        num_agents_same_direction, num_agents_opposite_direction = 0, 0
        distance_to_minor_node, distance_to_other_agent = np.inf, np.inf
        distance_to_own_target, distance_to_other_target = np.inf, np.inf
        min_agent_speed = 1.0

        # Skip ahead until we get to a major node, logging any agents on the tracks along the way
        while True:
            visited_cells.update(self.edge_paths[node.position, direction])
            visited_cells.add((*next_node.position, 0))

            if self.edge_paths[node.position, direction]:
                  row, column, dir = self.edge_paths[node.position, direction][-1]
                  orientation = first(self.get_possible_transitions((row, column), dir))
            else: orientation = direction

            key = (*node.position, direction)
            num_malfunctions += sum(1 for d in self.edges_with_malfunctions.get(key, []) if total_distance + d > 0)
            num_agents_same_direction += sum(1 for d, _ in self.edges_with_agents_going.get(key, {}).values() if total_distance + d > 0)
            num_agents_opposite_direction += sum(1 for d, _ in self.edges_with_agents_coming.get(key, {}).values() if total_distance + d > 0)
            num_agents_same_direction += 1 if (*next_node.position, orientation) in self.nodes_with_agents_going else 0
            num_agents_opposite_direction += 1 if (*next_node.position, orientation) in self.nodes_with_agents_coming else 0

            agent_distances = [edge_length + d for d, _ in self.edges_with_agents_going.get(key, {}).values() if total_distance + edge_length + d > 0] + \
                              [edge_length + d for d, _ in self.edges_with_agents_coming.get(key, {}).values() if total_distance + edge_length + d > 0] + \
                              [edge_length + distance if (*next_node.position, orientation) in self.nodes_with_agents_going else np.inf] + \
                              [edge_length + distance if (*next_node.position, orientation) in self.nodes_with_agents_coming else np.inf]
            agent_speeds    = [s for d, s in self.edges_with_agents_going.get(key, {}).values() if total_distance + edge_length + d > 0] + \
                              [self.nodes_with_agents_going.get((*next_node.position, orientation), 1.0)]

            distance_to_other_agent = min(distance_to_other_agent, *agent_distances)
            min_agent_speed = min(min_agent_speed, *agent_speeds)

            if next_node.is_target:
                if self.is_own_target(agent, next_node):
                      distance_to_own_target = min(distance_to_own_target, edge_length + distance)
                else: distance_to_other_target = min(distance_to_other_target, edge_length + distance)

            node = next_node
            edge_length += distance

            if len(node.edges) == 1 and not self.is_own_target(agent, node): # This is a minor node, keep exploring
                  direction, (next_node, distance) = first(node.edges.items())
                  if not node.is_target:
                      distance_to_minor_node = min(distance_to_minor_node, edge_length)
            else: break

        # Create a new tree node and populate its children
        if depth < self.max_depth:
            children = { x: -np.inf for x in ACTIONS }
            if not self.is_own_target(agent, node):
                for direction in node.edges.keys():
                    children[get_action(orientation, direction)] = \
                        self.get_tree_branch(agent, node, direction, visited_cells, total_distance + edge_length, depth + 1)

        else: children = {}

        return Node(dist_own_target_encountered=total_distance + distance_to_own_target,
                    dist_other_target_encountered=total_distance + distance_to_other_target,
                    dist_other_agent_encountered=total_distance + distance_to_other_agent,
                    dist_potential_conflict=np.inf,
                    dist_unusable_switch=total_distance + distance_to_minor_node,
                    dist_to_next_branch=total_distance + edge_length,
                    dist_min_to_target=self.env.distance_map.get()[(agent.handle, *node.position, orientation)] or 0,
                    num_agents_same_direction=num_agents_same_direction,
                    num_agents_opposite_direction=num_agents_opposite_direction,
                    num_agents_malfunctioning=num_malfunctions,
                    speed_min_fractional=min_agent_speed,
                    num_agents_ready_to_depart=0,
                    childs=children)


    # Helper functions

    def get_possible_transitions(self, position, direction):
        return [i for i, allowed in enumerate(self.env.rail.get_transitions(*position, direction)) if allowed]

    def get_all_transitions(self, position):
        bit_groups = f'{self.env.rail.get_full_transitions(*position):019_b}'.split("_")
        return [tuple(i for i, allowed in enumerate(bits) if allowed == '1') for bits in bit_groups]

    def is_junction(self, position):
        return any(len(transitions) > 1 for transitions in self.get_all_transitions(position))

    def is_target(self, position):
        return position in self.target_positions

    def is_own_target(self, agent, node):
        return agent.target == node.position

    def set_env(self, env):
        super().set_env(env)
        self.original_obs.set_env(env)
