from typing import List, Union, Tuple
import queue
from src.Grid import Grid


def shortest_swap_sequence_nn(grid_a: Grid, grid_b: Grid) -> Union[None, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Finds the shortest sequence of nearest-neighbor swaps to transform grid_a into grid_b.
    Uses a breadth-first search (BFS) algorithm.

    Args:
        grid_a: The starting Grid object.
        grid_b: The target Grid object.

    Returns:
        A list of swaps representing the shortest transformation sequence if a
        solution exists, otherwise None. Each swap is a tuple of coordinate pairs:
        ((row_1, col_1), (row_2, col_2)).
    """

    if grid_a == grid_b:
        return []  # No swaps needed if grids are already identical

    if grid_a.number_of_excitations() != grid_b.number_of_excitations():
        raise ValueError('The grids must have the same number of ones')

    if (grid_a.rows, grid_a.cols) != (grid_b.rows, grid_b.cols):
        raise ValueError('The grids must have the same shape')

    visited = set()  # Track visited grid configurations
    q = queue.Queue()  # Queue for the BFS exploration
    q.put((grid_a.copy(), []))  # Start with a copy of grid_a and empty swap list

    while not q.empty():
        current_grid, swaps = q.get()
        visited.add(current_grid.grid.tobytes())  # Mark the configuration as visited

        if current_grid == grid_b:
            return swaps  # Solution found

        # Generate all possible neighbors (using nearest-neighbor swaps)
        for r in range(current_grid.rows):
            for c in range(current_grid.cols):
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Offsets for neighbors
                    new_r, new_c = r + dr, c + dc
                    if 0 <= new_r < current_grid.rows and 0 <= new_c < current_grid.cols:  # Check in-bounds
                        neighbor_grid = current_grid.copy()  # Create a copy for the swap
                        neighbor_grid.swap((r, c), (new_r, new_c))  # Perform the swap

                        if neighbor_grid.grid.tobytes() not in visited:  # Check if unvisited
                            q.put((neighbor_grid, swaps + [((r, c), (new_r, new_c))]))  # Add to queue

    return None  # No solution found if the BFS doesn't find grid_b
