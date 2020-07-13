"""Rail generators (infrastructure manager, "Infrastrukturbetreiber")."""
import warnings
from typing import Callable, Tuple, Optional, Dict

cimport numpy as cnp
import numpy as np
from flatland.core.grid.grid4_utils import direction_to_point
from flatland.core.grid.grid_utils import Vec2dOperations
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map, connect_straight_line_in_grid_map, \
    fix_inner_nodes, align_cell_to_city
from numpy.random.mtrand import RandomState

RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]
cnp.import_array()
# CONSTANTS
cdef bint grid_mode = False
cdef int max_rails_between_cities = 3
cdef int max_rails_in_city = 4

cdef int NORTH = 0
cdef int EAST = 1
cdef int SOUTH = 2
cdef int WEST = 3

def generator(int width, int num_agents):
    cdef int city_padding = 2
    cdef int max_num_cities = max(2, width ** 2 // 300)

    rail_trans = RailEnvTransitions()
    grid_map = GridTransitionMap(width=width, height=width, transitions=rail_trans)
    # We compute the city radius by the given max number of rails it can contain.
    # The radius is equal to the number of tracks divided by 2
    # We add 2 cells to avoid that track lenght is to short
    # We use ceil if we get uneven numbers of city radius. This is to guarantee that all rails fit within the city.
    cdef int city_radius = ((max_rails_in_city + 1) // 2) + city_padding
    cdef cnp.ndarray vector_field = np.zeros(shape=(width, width)) - 1.


    # Calculate the max number of cities allowed
    # and reduce the number of cities to build to avoid problems
    cdef int max_feasible_cities = min(max_num_cities, ((width - 2) // (2 * (city_radius + 1))) ** 2)

    cdef bint too_close
    cdef int col, tries, row
    cdef tuple city_pos
    cdef list city_positions = []
    cdef int min_distance = (2 * (city_radius + 1) + 1)
    cdef int city_idx

    for city_idx in range(max_feasible_cities):
        too_close = True
        tries = 0

        while too_close:
            row = city_radius + 1 + np.random.randint(width - 2 * (city_radius + 1))
            col = city_radius + 1 + np.random.randint(width - 2 * (city_radius + 1))
            too_close = False
            # Check distance to cities
            for city_pos in city_positions:
                if np.abs(row - city_pos[0]) < min_distance and np.abs(col - city_pos[1]) < min_distance:
                    too_close = True
                    break

            if not too_close:
                city_positions.append((row, col))

            tries += 1
            if tries > 200:
                warnings.warn("Could not set all required cities!")
                break

    cdef list inner_connection_points = []
    cdef list outer_connection_points = []
    cdef list city_orientations = []
    cdef list city_cells = []
    cdef list neighb_dist, connection_sides_idx, connection_points_coordinates_outer
    cdef list connection_points_coordinates_inner, _city_cells
    cdef int current_closest_direction, idx, nr_of_connection_points
    cdef int number_of_out_rails, start_idx, direction, connection_idx
    cdef tuple neighbour_city, cell
    cdef tuple tmp_coordinates = tuple()
    cdef out_tmp_coordinates = tuple()
    cdef cnp.ndarray connections_per_direction, connection_slots, x_range, y_range, x_values, y_values, inner_point_offset
    for city_pos in city_positions:

        # Chose the directions where close cities are situated
        neighb_dist = []
        for neighbour_city in city_positions:
            neighb_dist.append(Vec2dOperations.get_manhattan_distance(city_pos, neighbour_city))
        closest_neighb_idx = np.argsort(neighb_dist)

        # Store the directions to these neighbours and orient city to face closest neighbour
        connection_sides_idx = []
        idx = 1
        current_closest_direction = direction_to_point(city_pos, city_positions[closest_neighb_idx[idx]])
        connection_sides_idx.append(current_closest_direction)
        connection_sides_idx.append((current_closest_direction + 2) % 4)
        city_orientations.append(current_closest_direction)
        x_range = np.arange(city_pos[0] - city_radius, city_pos[0] + city_radius + 1)
        y_range = np.arange(city_pos[1] - city_radius, city_pos[1] + city_radius + 1)
        x_values = np.repeat(x_range, len(y_range))
        y_values = np.tile(y_range, len(x_range))
        _city_cells = list(zip(x_values, y_values))
        for cell in _city_cells:
            vector_field[cell] = align_cell_to_city(city_pos, city_orientations[-1], cell)
        city_cells.extend(_city_cells)
        # set the number of tracks within a city, at least 2 tracks per city
        connections_per_direction = np.zeros(4, dtype=int)
        nr_of_connection_points = np.random.randint(2, max_rails_in_city + 1)
        for idx in connection_sides_idx:
            connections_per_direction[idx] = nr_of_connection_points
        connection_points_coordinates_inner = [[] for _ in range(4)]
        connection_points_coordinates_outer = [[] for _ in range(4)]
        number_of_out_rails = np.random.randint(1, min(max_rails_in_city, nr_of_connection_points) + 1)
        start_idx = int((nr_of_connection_points - number_of_out_rails) / 2)
        for direction in range(4):
            connection_slots = np.arange(nr_of_connection_points) - start_idx
            # Offset the rails away from the center of the city
            offset_distances = np.arange(nr_of_connection_points) - int(nr_of_connection_points / 2)
            # The clipping helps ofsetting one side more than the other to avoid switches at same locations
            # The magic number plus one is added such that all points have at least one offset
            inner_point_offset = np.abs(offset_distances) + np.clip(offset_distances, 0, 1) + 1
            for connection_idx in range(connections_per_direction[direction]):
                if direction == 0:
                    tmp_coordinates = (
                        city_pos[0] - city_radius + inner_point_offset[connection_idx],
                        city_pos[1] + connection_slots[connection_idx])
                    out_tmp_coordinates = (
                        city_pos[0] - city_radius, city_pos[1] + connection_slots[connection_idx])
                if direction == 1:
                    tmp_coordinates = (
                        city_pos[0] + connection_slots[connection_idx],
                        city_pos[1] + city_radius - inner_point_offset[connection_idx])
                    out_tmp_coordinates = (
                        city_pos[0] + connection_slots[connection_idx], city_pos[1] + city_radius)
                if direction == 2:
                    tmp_coordinates = (
                        city_pos[0] + city_radius - inner_point_offset[connection_idx],
                        city_pos[1] + connection_slots[connection_idx])
                    out_tmp_coordinates = (
                        city_pos[0] + city_radius, city_pos[1] + connection_slots[connection_idx])
                if direction == 3:
                    tmp_coordinates = (
                        city_pos[0] + connection_slots[connection_idx],
                        city_pos[1] - city_radius + inner_point_offset[connection_idx])
                    out_tmp_coordinates = (
                        city_pos[0] + connection_slots[connection_idx], city_pos[1] - city_radius)
                connection_points_coordinates_inner[direction].append(tmp_coordinates)
                if connection_idx in range(start_idx, start_idx + number_of_out_rails):
                    connection_points_coordinates_outer[direction].append(out_tmp_coordinates)

        inner_connection_points.append(connection_points_coordinates_inner)
        outer_connection_points.append(connection_points_coordinates_outer)

    cdef list inter_city_lines = []
    cdef list city_distances, closest_neighbours
    cdef int current_city_idx, direction_to_neighbour, out_direction, neighbour_idx

    for current_city_idx in np.arange(len(city_positions)):
        city_distances = []
        closest_neighbours = [None for _ in range(4)]

        # compute distance to all other cities
        for city_idx in range(len(city_positions)):
            city_distances.append(
                Vec2dOperations.get_manhattan_distance(city_positions[current_city_idx], city_positions[city_idx]))
        sorted_neighbours = np.argsort(city_distances)

        for neighbour in sorted_neighbours[1:]:  # do not include city itself
            direction_to_neighbour = direction_to_point(city_positions[current_city_idx], city_positions[neighbour])
            if closest_neighbours[direction_to_neighbour] is None:
                closest_neighbours[direction_to_neighbour] = neighbour

            # early return once all 4 directions have a closest neighbour
            if None not in closest_neighbours:
                break
        for out_direction in range(4):
            if closest_neighbours[out_direction] is not None:
                neighbour_idx = closest_neighbours[out_direction]
            elif closest_neighbours[(out_direction - 1) % 4] is not None:
                neighbour_idx = closest_neighbours[(out_direction - 1) % 4]  # counter-clockwise
            elif closest_neighbours[(out_direction + 1) % 4] is not None:
                neighbour_idx = closest_neighbours[(out_direction + 1) % 4]  # clockwise
            elif closest_neighbours[(out_direction + 2) % 4] is not None:
                neighbour_idx = closest_neighbours[(out_direction + 2) % 4]

            for city_out_connection_point in outer_connection_points[current_city_idx][out_direction]:

                min_connection_dist = np.inf
                neighbour_connection_point = None
                for direction in range(4):
                    current_points = outer_connection_points[neighbour_idx][direction]
                    for tmp_in_connection_point in current_points:
                        tmp_dist = Vec2dOperations.get_manhattan_distance(city_out_connection_point,
                                                                          tmp_in_connection_point)
                        if tmp_dist < min_connection_dist:
                            min_connection_dist = tmp_dist
                            neighbour_connection_point = tmp_in_connection_point

                new_line = connect_rail_in_grid_map(grid_map, city_out_connection_point, neighbour_connection_point,
                                                    rail_trans, flip_start_node_trans=False,
                                                    flip_end_node_trans=False, respect_transition_validity=False,
                                                    avoid_rail=True,
                                                    forbidden_cells=city_cells)
                inter_city_lines.extend(new_line)

    # Build inner cities
    cdef int i, current_city, opposite_boarder
    cdef int boarder = 0
    cdef int track_id, track_nbr
    cdef list free_rails = [[] for _ in range(len(city_positions))]
    for current_city in range(len(city_positions)):

        # This part only works if we have keep same number of connection points for both directions
        # Also only works with two connection direction at each city
        for i in range(4):
            if len(inner_connection_points[current_city][i]) > 0:
                boarder = i
                break

        opposite_boarder = (boarder + 2) % 4
        nr_of_connection_points = len(inner_connection_points[current_city][boarder])
        number_of_out_rails = len(outer_connection_points[current_city][boarder])
        start_idx = (nr_of_connection_points - number_of_out_rails) // 2
        # Connect parallel tracks
        for track_id in range(nr_of_connection_points):
            source = inner_connection_points[current_city][boarder][track_id]
            target = inner_connection_points[current_city][opposite_boarder][track_id]
            current_track = connect_straight_line_in_grid_map(grid_map, source, target, rail_trans)
            free_rails[current_city].append(current_track)

        for track_id in range(nr_of_connection_points):
            source = inner_connection_points[current_city][boarder][track_id]
            target = inner_connection_points[current_city][opposite_boarder][track_id]

            # Connect parallel tracks with each other
            fix_inner_nodes(
                grid_map, source, rail_trans)
            fix_inner_nodes(
                grid_map, target, rail_trans)

            # Connect outer tracks to inner tracks
            if start_idx <= track_id < start_idx + number_of_out_rails:
                source_outer = outer_connection_points[current_city][boarder][track_id - start_idx]
                target_outer = outer_connection_points[current_city][opposite_boarder][track_id - start_idx]
                connect_straight_line_in_grid_map(grid_map, source, source_outer, rail_trans)
                connect_straight_line_in_grid_map(grid_map, target, target_outer, rail_trans)

    # Populate cities
    cdef int num_cities = len(city_positions)
    cdef list train_stations = [[] for _ in range(num_cities)]
    for current_city in range(len(city_positions)):
        for track_nbr in range(len(free_rails[current_city])):
            possible_location = free_rails[current_city][track_nbr][
                int(len(free_rails[current_city][track_nbr]) / 2)]
            train_stations[current_city].append((possible_location, track_nbr))

    # Fix all transition elements

    cdef cnp.ndarray rails_to_fix = np.zeros(3 * grid_map.height * grid_map.width * 2, dtype='int')
    cdef int rails_to_fix_cnt = 0
    cdef list cells_to_fix = city_cells + inter_city_lines
    cdef bint cell_valid
    for cell in cells_to_fix:
        cell_valid = grid_map.cell_neighbours_valid(cell, True)

        if not cell_valid:
            rails_to_fix[3 * rails_to_fix_cnt] = cell[0]
            rails_to_fix[3 * rails_to_fix_cnt + 1] = cell[1]
            rails_to_fix[3 * rails_to_fix_cnt + 2] = vector_field[cell]

            rails_to_fix_cnt += 1
    # Fix all other cells
    for idx in range(rails_to_fix_cnt):
        grid_map.fix_transitions((rails_to_fix[3 * idx], rails_to_fix[3 * idx + 1]), rails_to_fix[3 * idx + 2])

    return grid_map, {'agents_hints': {
        'num_agents': num_agents,
        'city_positions': city_positions,
        'train_stations': train_stations,
        'city_orientations': city_orientations
    }}
