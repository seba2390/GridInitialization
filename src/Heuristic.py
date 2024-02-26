import numpy as np


def count_line_intersections(start, end, obstacle_cells):
    """
    Counts intersections of a line with a set of obstacle coordinates using Bresenham's line algorithm.

    This function implements Bresenham's line algorithm, an efficient method for
    determining the integer grid cells that a line passes through.  At each step of
    the algorithm, it checks if the current grid cell is an obstacle.

    Args:
        start: A tuple representing the starting coordinates (x0, y0) of the line.
        end: A tuple representing the ending coordinates (x1, y1) of the line.
        obstacle_cells: A NumPy array where each row represents an obstacle's
                        coordinate (x, y).

    Returns:
        The number of intersections between the line and the obstacles.
    """
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    intersection_count = 0
    while True:
        if (x0, y0) in obstacle_cells:  # Check if current cell is an obstacle
            intersection_count += 1

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return intersection_count


def heuristic_nn_distance(grid_a, grid_b, detour_factor=2):
    """
    Approximates the shortest nearest-neighbor swap sequence length considering obstacles.

    This heuristic builds upon the Manhattan distance by incorporating a penalty for
    obstacles. The steps are:
        1. Calculate the sum of Manhattan distances between mismatched excitations.
        2. Identify obstacles (fixed cells) in the grid.
        3. For each excitation pair:
            * Draw an ideal straight line between them.
            * Count intersections with obstacles.
            * Add a penalty to the distance based on the obstacle count  and detour_factor.

    Args:
        grid_a: The first NumPy array grid.
        grid_b: The second NumPy array grid.
        detour_factor: A multiplier controlling the estimated detour length
                       around obstacles.

    Returns:
        The approximated length of the shortest NN swap sequence.
    """

    if grid_a.shape != grid_b.shape:
        raise ValueError("Grids must have the same shape.")

    excitation_coords_a = np.argwhere(grid_a == 1)
    excitation_coords_b = np.argwhere(grid_b == 1)

    if excitation_coords_a.shape[0] != excitation_coords_b.shape[0]:
        raise ValueError("Grids must have the same number of excitations.")

    obstacle_cells = np.argwhere(grid_a == '#')  # Assuming '#' is an obstacle

    initial_distance = 0
    for i in range(len(excitation_coords_a)):
        distance = np.sum(np.abs(excitation_coords_a[i] - excitation_coords_b[i]))
        obstacle_count = count_line_intersections(excitation_coords_a[i], excitation_coords_b[i], obstacle_cells)
        initial_distance += distance + (obstacle_count * detour_factor)

    return initial_distance
