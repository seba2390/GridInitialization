import pytest
import numpy as np

from src.Grid import Grid  # Assuming your Grid class is in a file named grid.py


# Test Initialization
def test_init_with_dimensions():
    grid = Grid(dimensions=(3, 4))
    assert grid.rows == 3
    assert grid.cols == 4
    assert np.all(grid.grid == 0)


def test_init_with_size():
    grid = Grid(size=5)
    assert grid.rows == 5
    assert grid.cols == 5
    assert np.all(grid.grid == 0)


def test_init_without_arguments():
    with pytest.raises(ValueError):
        Grid()


def test_init_with_both_arguments():
    with pytest.raises(ValueError):
        Grid(dimensions=(2, 2), size=3)


# Test Equality (__eq__)
def test_equality_same_grid():
    grid1 = Grid(dimensions=(2, 2))
    grid2 = grid1
    assert grid1 == grid2


def test_equality_different_dimensions():
    grid1 = Grid(dimensions=(2, 2))
    grid2 = Grid(dimensions=(3, 2))
    assert grid1 != grid2


def test_equality_different_values():
    grid1 = Grid(dimensions=(2, 2))
    grid2 = Grid(dimensions=(2, 2))
    grid2.set(1, 0)
    assert grid1 != grid2


def test_equality_different_types():
    grid = Grid(dimensions=(2, 2))
    assert grid != [1, 2, 3, 4]  # Not a Grid object


# Test copy()
def test_copy():
    grid1 = Grid(dimensions=(3, 3))
    grid1.set(1, 1)
    grid2 = grid1.copy()

    assert grid1 == grid2  # Content should be equal
    grid2.set(0, 0)  # Modifying copy shouldn't affect original
    assert grid1 != grid2


# Test number_of_excitations()
def test_number_of_excitations_empty():
    grid = Grid(dimensions=(3, 3))
    assert grid.number_of_excitations() == 0


def test_number_of_excitations():
    grid = Grid(dimensions=(2, 3))
    grid.set(0, 0)
    grid.set(1, 1)
    assert grid.number_of_excitations() == 2


# Test set()
def test_set_valid():
    grid = Grid(dimensions=(3, 3))
    grid.set(1, 2)
    assert grid.grid[1, 2] == 1


def test_set_out_of_bounds():
    grid = Grid(dimensions=(3, 3))
    with pytest.raises(ValueError):
        grid.set(4, 2)


# Test swap()
def test_swap_valid():
    grid = Grid(dimensions=(3, 3))
    grid.set(0, 1)  # Set value of 1 at (0, 1)
    grid.swap((0, 1), (2, 0))
    grid.swap((0, 1), (2, 0))

    assert grid.grid[0, 1] == 1  # Should be 1 after swap
    assert grid.grid[2, 0] == 0  # Should be 0 after swap


def test_swap_out_of_bounds():
    grid = Grid(dimensions=(3, 3))
    with pytest.raises(ValueError):
        grid.swap((2, 1), (3, 0))  # One position is out of bounds


# Test set_configuration()
def test_set_configuration_valid():
    grid = Grid(dimensions=(3, 3))
    new_config = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
    grid.set_configuration(new_config)
    assert np.array_equal(grid.grid, new_config)


def test_set_configuration_invalid_type():
    grid = Grid(dimensions=(2, 2))
    with pytest.raises(ValueError):
        grid.set_configuration([1, 0, 1, 0])  # Not a NumPy array


def test_set_configuration_mismatch_shape():
    grid = Grid(dimensions=(2, 2))
    with pytest.raises(ValueError):
        grid.set_configuration(np.array([[0, 1]]))  # Wrong shape


def test_set_configuration_invalid_values():
    grid = Grid(dimensions=(2, 2))
    with pytest.raises(ValueError):
        grid.set_configuration(np.array([[0, 2, 1, 0]]))  # Contains a value other than 0 or 1