import pytest
from src.Grid import Grid
from src.BFS import shortest_swap_sequence_nn


def test_identical_grids():
    grid_a = Grid(dimensions=(3, 3))
    grid_b = grid_a.copy()

    swaps = shortest_swap_sequence_nn(grid_a, grid_b)
    assert swaps == []


def test_different_shapes():
    grid_a = Grid(dimensions=(3, 3))
    grid_b = Grid(dimensions=(3, 4))

    with pytest.raises(ValueError):
        shortest_swap_sequence_nn(grid_a, grid_b)


def test_different_number_of_excitations():
    grid_a = Grid(dimensions=(3, 3))
    grid_b = Grid(dimensions=(3, 3))
    grid_b.set(0, 0)  # Different number of excitations

    with pytest.raises(ValueError):
        shortest_swap_sequence_nn(grid_a, grid_b)


def test_valid_swap_sequence():
    grid_a = Grid(dimensions=(2, 2))
    grid_b = grid_a.copy()
    grid_b.set(0, 1)
    grid_b.set(1, 0)

    swaps = shortest_swap_sequence_nn(grid_a, grid_b)
    expected_swaps = [((0, 0), (1, 1)), ((0, 1), (1, 0))]

    assert swaps == expected_swaps


def test_nn_swap_simple_solution():
    grid_a = Grid(dimensions=(3, 3))
    grid_b = Grid(dimensions=(3, 3))
    grid_a.set(0, 0)
    grid_b.set(0, 1)

    swap_sequence = shortest_swap_sequence_nn(grid_a, grid_b)
    assert swap_sequence == [((0, 0), (0, 1))]  # Should be a single swap


def test_nn_swap_multiple_steps():
    grid_a = Grid(dimensions=(3, 3))
    grid_b = Grid(dimensions=(3, 3))
    grid_a.set(0, 0)
    grid_b.set(2, 0)

    swap_sequence = shortest_swap_sequence_nn(grid_a, grid_b)
    assert len(swap_sequence) == 2  # Check length, exact sequence might vary


def test_bfs_nn():
    for repetition in range(10):
        grid_a = Grid(dimensions=(3, 3))
        grid_a.set_random_configuration(n_excitations=3)

        grid_b = Grid(dimensions=(3, 3))
        grid_b.set_random_configuration(n_excitations=3)

        for swap in shortest_swap_sequence_nn(grid_a,grid_b):
            grid_a.swap(swap[0], swap[1])

        assert grid_a == grid_b