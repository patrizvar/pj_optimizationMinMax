import random
from typing import Tuple, Optional
import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    moves = []

    for col in range(board.shape[1]):
        if 0 in board[:, col]: # find zeros in column and add them to the list
            moves.append(col)
    if moves:
        action = random.choice(moves) # choose a random column from the list
    else:
        action = None
    if action is not None:
        return action, saved_state

    raise ValueError("moves unavailable.")

