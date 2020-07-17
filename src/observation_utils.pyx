from collections import defaultdict
cimport numpy as cnp
import numpy as np
import torch
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv

cdef list ACTIONS = ['L', 'F', 'R', 'B']
Node = TreeObsForRailEnv.Node

cdef int positive_infinity = int(1e5)
cdef int negative_infinity = -positive_infinity

def first(iterable):
    for elem in iterable:
        return elem

cpdef bint _check_len1(tuple obj):
    return len(obj) > 1

cpdef str get_action(int orientation, int direction):
    return ACTIONS[(direction - orientation + 1 + 4) % 4]

cpdef int get_direction(int orientation, int action):
    if action == 1:
        return (orientation + 4 - 1) % 4
    elif action == 3:
        return (orientation + 1) % 4
    else:
        return orientation

cdef class RailNode:
    cdef public dict edges
    cdef public tuple position
    cdef public tuple edge_directions
    cdef public int is_target
    def __init__(self, tuple position, tuple edge_directions, int is_target):
        self.edges = {}
        self.position = position
        self.edge_directions = edge_directions
        self.is_target = is_target

    def __repr__(self):
        return f'RailNode({self.position}, {len(self.edges)})'


class GlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),\
          assuming 16 bits encoding of transitions.

        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
            - fifth channel containing number of other agents ready to depart

        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """
    def __init__(self):
        super(GlobalObsForRailEnv, self).__init__()
        self.size = 0
        self._custom_rail_obs = None
    def reset(self):
        if self._custom_rail_obs is None:
            self._custom_rail_obs = np.zeros((1, self.env.height + 2*self.size, self.env.width + 2*self.size, 16))

        self._custom_rail_obs[0, self.size:-self.size, self.size:-self.size] = np.array([[[[1 if digit == '1' else 0
                                                                                    for digit in
                                                                                    f'{self.env.rail.get_full_transitions(i, j):016b}']
                                                                                   for j in range(self.env.width)]
                                                                                  for i in range(self.env.height)]],
                                                                                dtype=np.float32)

    def get_many(self, list trash):
        cdef int agent_count = len(self.env.agents)
        cdef cnp.ndarray obs_agents_state = np.zeros((agent_count,
                                                     self.env.height,
                                                     self.env.width,
                                                     5), dtype=np.float32)
        cdef int i, agent_id
        cdef tuple pos, agent_virtual_position
        for agent_id, agent in enumerate(self.env.agents):
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_virtual_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                agent_virtual_position = agent.target
            else:
                continue

            obs_agents_state[agent_id, :, :, 0:4] = -1

            obs_agents_state[(agent_id,) + agent_virtual_position + (0,)] = agent.direction

            for i, other_agent in enumerate(self.env.agents):

                # ignore other agents not in the grid any more
                if other_agent.status == RailAgentStatus.DONE_REMOVED:
                    continue

                # second to fourth channel only if in the grid
                if other_agent.position is not None:
                    pos = (agent_id,) + other_agent.position
                    # second channel only for other agents
                    if i != agent_id:
                        obs_agents_state[pos + (1,)] = other_agent.direction
                    obs_agents_state[pos + (2,)] = other_agent.malfunction_data['malfunction']
                    obs_agents_state[pos + (3,)] = other_agent.speed_data['speed']
                # fifth channel: all ready to depart on this position
                if other_agent.status == RailAgentStatus.READY_TO_DEPART:
                    obs_agents_state[(agent_id,) + other_agent.initial_position + (4,)] += 1
        return {i: arr
                for i, arr in
                enumerate(np.concatenate([np.repeat(self.rail_obs, agent_count, 0), obs_agents_state], -1))}


class LocalObsForRailEnv(GlobalObsForRailEnv):
    def __init__(self, size=7):
        super(LocalObsForRailEnv, self).__init__()
        self.size = size
    def get_many(self, list trash):
        cdef int agent_count = len(self.env.agents)
        obs_agents_state = np.zeros((agent_count,
                                                     self.size * 2 + 1,
                                                     self.size * 2 + 1,
                                                     21), dtype=np.float32)
        cdef int i, agent_id
        cdef tuple agent_virtual_position
        for agent_id, agent in enumerate(self.env.agents):
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_virtual_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                agent_virtual_position = agent.target
            else:
                continue
            x0, y0, x1, y1 = (agent_virtual_position[0],
                              agent_virtual_position[1],
                              agent_virtual_position[0] + 2*self.size + 1,
                              agent_virtual_position[1] + 2*self.size + 1)
            obs_agents_state[agent_id, :, :, 5:] = self._custom_rail_obs[0, x0:x1, y0:y1]

            obs_agents_state[agent_id, :, :, 0:4] = -1

            obs_agents_state[agent_id, :, :, 0] = agent.direction

            for i, other_agent in enumerate(self.env.agents):

                # ignore other agents not in the grid any more
                if other_agent.status == RailAgentStatus.DONE_REMOVED:
                    continue

                # second to fourth channel only if in the grid
                if other_agent.position is not None:
                    pos = (agent_id,) + other_agent.position
                    # second channel only for other agents
                    if i != agent_id:
                        obs_agents_state[agent_id, :, :, 1] = other_agent.direction
                    obs_agents_state[agent_id, :, :, 2] = other_agent.malfunction_data['malfunction']
                    obs_agents_state[agent_id, :, :, 3] = other_agent.speed_data['speed']
                # fifth channel: all ready to depart on this position
                if other_agent.status == RailAgentStatus.READY_TO_DEPART:
                    init_pos = other_agent.initial_position
                    dist0 = agent_virtual_position[0] - init_pos[0]
                    dist1 = agent_virtual_position[1] - init_pos[1]
                    if abs(dist0) < self.size and abs(dist1) < self.size:
                        obs_agents_state[agent_id, dist0 + self.size, dist1 + self.size, 4] += 1
        return {i: arr
                for i, arr in
                enumerate(obs_agents_state)}


class TreeObservation(ObservationBuilder):
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 11

    # Create a graph representation of the current rail network

    def reset(self):
        self.target_positions = {agent.target: 1 for agent in self.env.agents}
        self.edge_positions = defaultdict(list)  # (cell.position, direction) -> [(start, end, direction, distance)]
        self.edge_paths = defaultdict(list)  # (node.position, direction) -> [(cell.position, direction)]

        # First, we find a node by starting at one of the agents and following the rails until we reach a junction
        agent = first(self.env.agents)
        cpdef tuple position = tuple(agent.initial_position)
        cpdef int direction = agent.direction
        cdef bint out
        while True:
            try:
                out = self.is_junction(position) or self.is_target(position)
            except IndexError:
                break
            if not out:
                break
            direction = first(self.get_possible_transitions(position, direction))
            position = get_new_position(position, direction)

        # Now we create a graph representation of the rail network, starting from this node
        cdef tuple transitions = self.get_all_transitions(position)
        cdef dict root_nodes = {t: RailNode(position, t, self.is_target(position)) for t in transitions if t}
        self.graph = {(*position, d): root_nodes[t] for d, t in enumerate(transitions) if t}

        for transitions, node in root_nodes.items():
            for direction in transitions:
                self.explore_branch(node, get_new_position(position, direction), direction)

    def explore_branch(self, RailNode node, tuple position, int direction):
        cdef int original_direction = direction
        cdef dict edge_positions = {}
        cdef int distance = 1
        cdef int next_direction = 0
        cdef int idx = 0
        cdef tuple transition = tuple()
        cdef tuple key = tuple()

        # Explore until we find a junction
        while not self.is_junction(position) and not self.is_target(position):
            next_direction = first(self.get_possible_transitions(position, direction))
            edge_positions[(*position, direction)] = (distance, next_direction)
            position = get_new_position(position, next_direction)
            direction = next_direction
            distance += 1

        # Create any nodes that aren't in the graph yet
        cdef tuple transitions = self.get_all_transitions(position)
        cdef bint is_target = self.is_target(position)
        cdef dict nodes = {transition: RailNode(position, transition, is_target)
                           for idx, transition in enumerate(transitions)
                           if transition and (*position, idx) not in self.graph}

        for idx, transition in enumerate(transitions):
            if transition in nodes:
                self.graph[(*position, idx)] = nodes[transition]

        # Connect the previous node to the next one, and update self.edge_positions
        cdef RailNode next_node = self.graph[(*position, direction)]
        node.edges[original_direction] = (next_node, distance)
        for key, (distance, next_direction) in edge_positions.items():
            self.edge_positions[key].append((node, next_node, original_direction, distance))
            self.edge_paths[node.position, original_direction].append((*key, next_direction))

        # Call ourselves recursively since we're exploring depth-first
        for transitions, node in nodes.items():
            for direction in transitions:
                self.explore_branch(node, get_new_position(position, direction), direction)

    # Create a tree observation for each agent, based on the graph we created earlier

    def get_many(self, list handles=[]):
        self.nodes_with_agents_going, self.edges_with_agents_going = {}, defaultdict(dict)
        self.nodes_with_agents_coming, self.edges_with_agents_coming = {}, defaultdict(dict)
        self.nodes_with_malfunctions, self.edges_with_malfunctions = {}, defaultdict(dict)
        self.nodes_with_departures, self.edges_with_departures = {}, defaultdict(dict)

        cdef int direction = 0

        # Create some lookup tables that we can use later to figure out how far away the agents are from each other.
        for agent in self.env.agents:
            if agent.status == RailAgentStatus.READY_TO_DEPART and agent.initial_position:
                for direction in range(4):
                    if (*agent.initial_position, direction) in self.graph:
                        self.nodes_with_departures[(*agent.initial_position, direction)] = 1

                    for start, _, start_direction, distance in self.edge_positions[
                        (*agent.initial_position, direction)]:
                        self.edges_with_departures[(*start.position, start_direction)][agent.handle] = distance

            if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
                agent_key = (*agent.position, agent.direction)
                for direction in range(4):
                    # # Check the nodes
                    if (*agent.position, direction) in self.graph:
                        node_dict = self.nodes_with_agents_going if direction == agent.direction else self.nodes_with_agents_coming
                        node_dict[(*agent.position, direction)] = agent.speed_data['speed']

                        # if len(self.graph[agent_key].edges) > 1:
                        #     exit_direction = get_direction(agent.direction, agent.speed_data['transition_action_on_cellexit'])
                        #     if agent.speed_data['position_fraction'] == 0 or exit_direction not in self.graph[agent_key].edges: # Agent still has options
                        #         self.nodes_with_agents_going[(*agent.position, direction)] = agent.speed_data['speed']
                        #     else: # Agent has already decided
                        #         coming_direction = (exit_direction + 2) % 4
                        #         node_dict = self.nodes_with_agents_coming if direction == coming_direction else self.nodes_with_agents_going
                        #         node_dict[(*agent.position, direction)] = agent.speed_data['speed']
                        # else:
                        #     exit_direction = first(self.graph[agent_key].edges.keys())
                        #     coming_direction = (exit_direction + 2) % 4
                        #     node_dict = self.nodes_with_agents_coming if direction == coming_direction else self.nodes_with_agents_going
                        #     node_dict[(*agent.position, direction)] = agent.speed_data['speed']

                    # Check the edges
                    if agent_key in self.edge_positions:
                        exit_direction = first(self.get_possible_transitions(agent.position, agent.direction))
                        coming_direction = (exit_direction + 2) % 4
                        edge_dict = self.edges_with_agents_coming if direction == coming_direction else self.edges_with_agents_going
                        if direction == agent.direction or direction == coming_direction:
                            for start, _, start_direction, distance in self.edge_positions[
                                (*agent.position, direction)]:
                                edge_dict[(*start.position, start_direction)][agent.handle] = (
                                    distance, agent.speed_data['speed'])

                    # Check for malfunctions
                    if agent.malfunction_data['malfunction']:
                        if (*agent.position, direction) in self.graph:
                            self.nodes_with_malfunctions[(*agent.position, direction)] = agent.malfunction_data[
                                'malfunction']

                        for start, _, start_direction, distance in self.edge_positions[(*agent.position, direction)]:
                            self.edges_with_malfunctions[(*start.position, start_direction)][agent.handle] = \
                                (distance, agent.malfunction_data['malfunction'])

        return super().get_many(handles)

    # Compute the observation for a single agent
    def get(self, int handle):
        agent = self.env.agents[handle]
        cdef set visited_cells = set()

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_position = agent.target
        else:
            return None

        # The root node contains information about the agent itself
        cdef int direction = 0
        cdef int distance = 0
        root_tree_node = Node(0, 0, 0, 0, 0, 0, self.env.distance_map.get()[(handle, *agent_position, agent.direction)],
                              0, 0, agent.malfunction_data['malfunction'], agent.speed_data['speed'], 0,
                              {x: negative_infinity for x in ACTIONS})

        # Then we build out the tree by exploring from this node
        cdef tuple key = (*agent_position, agent.direction)
        cdef RailNode node = RailNode(tuple(), tuple(), 0)
        cdef RailNode prev_node = RailNode(tuple(), tuple(), 0)
        if key in self.graph:  # If we're sitting on a junction, branch out immediately
            node = self.graph[key]
            if len(node.edges) > 1:  # Major node
                for direction in self.graph[key].edges.keys():
                    root_tree_node.childs[get_action(agent.direction, direction)] = \
                        self.get_tree_branch(agent, node, direction, visited_cells, 0, 1)
            else:  # Minor node
                direction = first(self.get_possible_transitions(node.position, agent.direction))
                root_tree_node.childs['F'] = self.get_tree_branch(agent, node, direction, visited_cells, 0, 1)

        else:  # Just create a single child in the forward direction
            prev_node, _, direction, distance = first(self.edge_positions[key])
            root_tree_node.childs['F'] = self.get_tree_branch(agent, prev_node, direction, visited_cells, -distance, 1)

        self.env.dev_obs_dict[handle] = visited_cells

        return root_tree_node

    # Get the next tree node, starting from `node`, facing `orientation`, and moving in `direction`.
    def get_tree_branch(self, agent, RailNode node, int direction, visited_cells, int total_distance, int depth):
        visited_cells.add((*node.position, 0))
        next_node, distance = node.edges[direction]

        cdef int edge_length = 0
        cdef int max_malfunction_length = 0
        cdef int num_agents_same_direction = 0
        cdef int num_agents_other_direction = 0
        cdef int distance_to_minor_node = positive_infinity
        cdef int distance_to_other_agent = positive_infinity
        cdef int distance_to_own_target = positive_infinity
        cdef int distance_to_other_target = positive_infinity
        cdef float min_agent_speed = 1
        cdef int num_agent_departures = 0

        cdef int orientation = 0
        cdef int dist = 0

        cdef int tmp_dist = 0
        cdef int tmp1 = 0  # Speed/Malfunction Length

        cdef tuple key = tuple()
        cdef tuple next_key = tuple()

        cdef list path = list()

        # Skip ahead until we get to a major node, logging any agents on the tracks along the way
        while True:
            path = self.edge_paths.get((node.position, direction), [])
            orientation = path[-1][-1] if path else direction
            dist = total_distance + edge_length
            key = (*node.position, direction)
            next_key = (*next_node.position, orientation)

            visited_cells.update(path)
            visited_cells.add((*next_node.position, 0))

            # Check for other agents on the junctions up ahead
            if next_key in self.nodes_with_agents_going:
                num_agents_same_direction += 1
                # distance_to_other_agent = min(distance_to_other_agent, edge_length + distance)
                min_agent_speed = min(min_agent_speed, self.nodes_with_agents_going[next_key])

            if next_key in self.nodes_with_agents_coming:
                num_agents_other_direction += 1
                distance_to_other_agent = min(distance_to_other_agent, edge_length + distance)

            if next_key in self.nodes_with_departures:
                num_agent_departures += 1
            if next_key in self.nodes_with_malfunctions:
                max_malfunction_length = max(max_malfunction_length, self.nodes_with_malfunctions[next_key])

            # Check for other agents along the tracks up ahead
            for tmp_dist, tmp1 in self.edges_with_agents_going[key].values():
                if dist + tmp_dist > 0:
                    num_agents_same_direction += 1
                    min_agent_speed = min(min_agent_speed, tmp1)
                    # distance_to_other_agent = min(distance_to_other_agent, edge_length + d)

            for tmp_dist, _ in self.edges_with_agents_coming[key].values():
                if dist + tmp_dist > 0:
                    num_agents_other_direction += 1
                    distance_to_other_agent = min(distance_to_other_agent, edge_length + tmp_dist)

            for tmp_dist in self.edges_with_departures[key].values():
                if dist + tmp_dist > 0:
                    num_agent_departures += 1

            for tmp_dist, tmp1 in self.edges_with_malfunctions[key].values():
                if dist + tmp_dist > 0:
                    max_malfunction_length = max(max_malfunction_length, tmp1)

            # Check for target nodes up ahead
            if next_node.is_target:
                if self.is_own_target(agent, next_node):
                    distance_to_own_target = min(distance_to_own_target, edge_length + distance)
                else:
                    distance_to_other_target = min(distance_to_other_target, edge_length + distance)

            # Move on to the next node
            node = next_node
            edge_length += distance

            if len(node.edges) == 1 and not self.is_own_target(agent, node):  # This is a minor node, keep exploring
                direction, (next_node, distance) = first(node.edges.items())
                if not node.is_target:
                    distance_to_minor_node = min(distance_to_minor_node, edge_length)
            else:
                break

        # Create a new tree node and populate its children
        cdef dict children = {}
        cdef str x = ''
        if depth < self.max_depth:
            for x in ACTIONS:
                children[x] = negative_infinity
            if not self.is_own_target(agent, node):
                for direction in node.edges.keys():
                    children[get_action(orientation, direction)] = \
                        self.get_tree_branch(agent, node, direction, visited_cells, total_distance + edge_length,
                                             depth + 1)

        return Node(dist_own_target_encountered=total_distance + distance_to_own_target,
                    dist_other_target_encountered=total_distance + distance_to_other_target,
                    dist_other_agent_encountered=total_distance + distance_to_other_agent,
                    dist_potential_conflict=positive_infinity,
                    dist_unusable_switch=total_distance + distance_to_minor_node,
                    dist_to_next_branch=total_distance + edge_length,
                    dist_min_to_target=self.env.distance_map.get()[(agent.handle, *node.position, orientation)] or 0,
                    num_agents_same_direction=num_agents_same_direction,
                    num_agents_opposite_direction=num_agents_other_direction,
                    num_agents_malfunctioning=max_malfunction_length,
                    speed_min_fractional=min_agent_speed,
                    num_agents_ready_to_depart=num_agent_departures,
                    childs=children)

    # Helper functions

    def get_possible_transitions(self, tuple position, int direction):
        return [i for i, allowed in enumerate(self.env.rail.get_transitions(*position, direction)) if allowed]

    def get_all_transitions(self, tuple position):
        return tuple(tuple(i for i, allowed in enumerate(bits) if allowed == '1')
                     for bits in f'{self.env.rail.get_full_transitions(*position):019_b}'.split("_"))

    def is_junction(self, tuple position):
        return any(map(_check_len1, self.get_all_transitions(position)))

    def is_target(self, tuple position):
        return position in self.target_positions

    def is_own_target(self, agent, RailNode node):
        return agent.target == node.position


ZERO_NODE = torch.zeros((1, 11))

# Recursively create numpy arrays for each tree node
cpdef create_tree_features(node, int max_depth, list data):
    cdef list nodes = [(node, 0)]
    cdef int current_depth = 0
    for node, current_depth in nodes:
        if node == negative_infinity or node == positive_infinity or node is None:
            data.append(ZERO_NODE.expand((4 ** (max_depth - current_depth + 1) - 1) // 3, -1))
        else:
            data.append(torch.FloatTensor(node[:-2]).view(1, 11))
            if node.childs:
                for direction in ACTIONS:
                    nodes.append((node.childs[direction], current_depth + 1))

# Normalize a tree observation
cpdef normalize_observation(tuple observations, int max_depth, shared_tensor, int starting_index):
    cdef list data = []
    cdef int i = 0
    for i, tree in enumerate(observations, 1):
        if tree is None:
            break
        data.append([])
        if isinstance(tree, dict):
            tree = tree.values()
        for t in tree:
            data[-1].append([])
            if isinstance(t, dict):
                for d in t.values():
                    create_tree_features(d, max_depth, data[-1][-1])
            else:
                create_tree_features(t, max_depth, data[-1][-1])

    shared_tensor[starting_index:starting_index + i] = torch.stack([torch.stack([torch.cat(dat, 0)
                                                                                 for dat in tree if dat != []], -1)
                                                                    for tree in data if tree != []], 0)
