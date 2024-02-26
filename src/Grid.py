from typing import Tuple
import numpy as np
from src.Tools import generate_random_tuples


class Grid:
    def __init__(self, dimensions: Tuple[int, int] = None, size: int = None):
        if dimensions is None and size is None:
            raise ValueError('Either dimensions or size must be specified...')
        if dimensions is not None and size is not None:
            raise ValueError('Size of grid should specified either with dimensions or size (not both)...')
        self.rows, self.cols = dimensions if size is None else (size, size)
        self.grid = self._initialize_grid()

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False  # Not the same type, not equal
        # Check that dimensions match
        if self.rows != other.rows or self.cols != other.cols:
            return False
        # Compare grid elements
        return np.array_equal(self.grid, other.grid)

    def copy(self):
        new_grid = Grid(dimensions=(self.rows, self.cols))
        new_grid.grid = self.grid.copy()  # Copy the NumPy array
        return new_grid

    def _initialize_grid(self):
        return np.array([[0 for col in range(self.cols)] for row in range(self.rows)], dtype=np.int8)

    def show(self):
        print('-' * (2 * self.cols + 1))
        for row in range(self.rows):
            row_string = "|"
            for col in range(self.cols):
                row_string += str(self.grid[row, col]) + "|"
            print(row_string)
            print('-' * (2 * self.cols + 1))

    def number_of_excitations(self) -> int:
        return np.sum(self.grid)

    def set(self, row: int, column: int) -> None:
        if not (0 <= row < self.rows and 0 <= column < self.cols):
            raise ValueError('Row and column should be within the grid...')
        self.grid[row, column] = np.int8(1)

    def swap(self, position_1: Tuple[int, int], position_2: Tuple[int, int]) -> None:
        if not (0 <= position_1[0] < self.rows and 0 <= position_1[1] < self.cols):
            raise ValueError('Position 1 should be within the grid...')
        if not (0 <= position_2[0] < self.rows and 0 <= position_2[1] < self.cols):
            raise ValueError('Position 2 should be within the grid...')

        # Direct swap using tuple packing and unpacking
        self.grid[position_1[0], position_1[1]], self.grid[position_2[0], position_2[1]] = \
            self.grid[position_2[0], position_2[1]], self.grid[position_1[0], position_1[1]]

    def set_configuration(self, configuration: np.ndarray):
        if not isinstance(configuration, np.ndarray):
            raise ValueError("Invalid configuration type. Must be a NumPy array.")
        if configuration.shape != (self.rows, self.cols):
            raise ValueError("Invalid configuration shape. Must match the dimensions of the grid.")
        if not np.all(np.logical_or(configuration == 0, configuration == 1)):
            raise ValueError("Invalid configuration values. Must contain only 0 or 1.")
        self.grid = configuration

    def set_random_configuration(self, n_excitations: int):
        if not 1 <= n_excitations <= self.rows * self.cols:
            raise ValueError(f'N_excitations should be a positive integer between 1 and the number of entries in the '
                             f'grid, i.e. [1;{self.rows * self.cols}]')
        for (row, col) in generate_random_tuples(rows=self.rows,cols=self.cols, k=n_excitations):
            self.grid[row, col] = np.int8(1)

    def reset_configuration(self):
        self.grid *= np.int8(0)
