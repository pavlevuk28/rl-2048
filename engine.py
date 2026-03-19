from enum import Enum
import numpy as np
from collections import namedtuple, deque

# Config
P_NEW_TILE_IS_TWO = 0.9

# Some useful classes
Game_state = namedtuple(
    "Game_state", ("board", "new_board", "best_move", "implied_value_from_best_move")
)

class Move(Enum):
    U = "U"
    L = "L"
    R = "R"
    D = "D"

# mostly chatGPT generated engine code
def _compress_and_merge(row):
    """
    Returns (new_row, points_scored).
    Merging x + x produces x points.
    """
    row = row[row != 0]  # remove zeros
    merged = []
    points = 0

    skip = False
    for i in range(len(row)):
        if skip:
            skip = False
            continue

        if i + 1 < len(row) and row[i] == row[i + 1]:
            merged_val = row[i] * 2
            merged.append(merged_val)
            points += row[i]  # scoring rule: x + x -> x points
            skip = True
        else:
            merged.append(row[i])

    merged = np.array(merged, dtype=int)
    merged = np.pad(merged, (0, 4 - len(merged)))
    return merged, points


def apply_move(board: np.ndarray, move: Move):
    """
    Returns (new_board, points_scored, changed)
    changed=True iff the board is different after the move.
    """
    original = np.array(board, copy=True)
    b = np.array(board, copy=True)
    total_points = 0

    if move == Move.L:
        for i in range(4):
            b[i], pts = _compress_and_merge(b[i])
            total_points += pts

    elif move == Move.R:
        for i in range(4):
            reversed_row = b[i][::-1]
            merged, pts = _compress_and_merge(reversed_row)
            b[i] = merged[::-1]
            total_points += pts

    elif move == Move.U:
        b = b.T
        for i in range(4):
            b[i], pts = _compress_and_merge(b[i])
            total_points += pts
        b = b.T

    elif move == Move.D:
        b = b.T
        for i in range(4):
            reversed_col = b[i][::-1]
            merged, pts = _compress_and_merge(reversed_col)
            b[i] = merged[::-1]
            total_points += pts
        b = b.T

    changed = not np.array_equal(original, b)
    return b, total_points, changed


def add_new_tile(board: np.ndarray):
    """
    Selects one empty (zero) cell uniformly at random and
    inserts 2 with probability P_NEW_TILE_IS_TWO, else 4.
    Returns a new board.
    """
    b = np.array(board, copy=True)

    # Find all zero locations
    empty = np.argwhere(b == 0)
    if empty.size == 0:
        return b  # no space to add a tile

    # Pick a random empty position
    i, j = empty[np.random.randint(len(empty))]

    # Choose value
    value = 2 if np.random.rand() < P_NEW_TILE_IS_TWO else 4
    b[i, j] = value

    return b


def get_new_board():
    empty_board = np.zeros((4, 4), dtype=int)

    return add_new_tile(add_new_tile(empty_board))

# useful util for computing expectations based on next state
def possible_next_boards(board):
    """
    Returns a list of tuples: (board, probability)
    showing all possible boards which can result after a new tile is added
    """
    # Find indices of zeros
    zero_indices = list(zip(*np.where(board == 0)))
    n_zeros = len(zero_indices)

    assert n_zeros > 0

    # Prepare the list of possible boards
    next_boards = []

    for i, j in zero_indices:
        # New tile is 2
        b2 = board.copy()
        b2[i, j] = 2
        prob2 = P_NEW_TILE_IS_TWO / n_zeros
        next_boards.append((b2, prob2))

        # New tile is 4
        b4 = board.copy()
        b4[i, j] = 4
        prob4 = (1 - P_NEW_TILE_IS_TWO) / n_zeros
        next_boards.append((b4, prob4))

    return next_boards