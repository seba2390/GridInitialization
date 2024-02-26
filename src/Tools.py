from typing import List, Tuple
import random
import numpy as np


def generate_random_tuples(rows: int, cols: int, k: int) -> List[Tuple[int, int]]:
    """
    Generates 'k' random tuples, each containing two positive integers within the
    specified range, ensuring no duplicates.

    Args:
        rows: The upper bound (exclusive) for the first integer (row index).
        cols: The upper bound (exclusive) for the second integer (column index).
        k: The number of tuples to generate.

    Returns:
        A list of 'k' unique tuples, each containing two positive integers within
        the specified range.

    Raises:
        ValueError: If it's not possible to generate the requested number of tuples
                    without duplicates within the given grid dimensions.
    """

    if k > rows * cols:  # Check if it's even possible
        raise ValueError("Cannot generate more tuples than possible grid cells.")

    seen = set()  # Keep track of the generated tuples
    tuples = []
    while len(tuples) < k:
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)
        new_tuple = (row, col)
        if new_tuple not in seen:
            seen.add(new_tuple)
            tuples.append(new_tuple)

    return tuples


def manhattan_distance_grids(grid_a: np.ndarray, grid_b: np.ndarray) -> int:
    """
    Calculates the Manhattan distance between two grids. Assumes only excitations
    (values of '1') are relevant for the distance calculation.

    Args:
        grid_a: The first NumPy array grid.
        grid_b: The second NumPy array grid.

    Returns:
        The Manhattan distance between the grids.

    Raises:
        ValueError: If the grids have different shapes.
    """

    if grid_a.shape != grid_b.shape:
        raise ValueError("Grids must have the same shape.")

    # Find the coordinates of excitations (where the value is '1')
    excitation_coords_a = np.argwhere(grid_a == 1)
    excitation_coords_b = np.argwhere(grid_b == 1)

    # Ensure that there's a corresponding excitation in the other grid
    if excitation_coords_a.shape[0] != excitation_coords_b.shape[0]:
        raise ValueError("Grids must have the same number of excitations.")

    total_distance = 0
    for i in range(len(excitation_coords_a)):
        distance = np.sum(np.abs(excitation_coords_a[i] - excitation_coords_b[i]))
        total_distance += distance

    return total_distance