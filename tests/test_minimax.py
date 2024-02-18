import numpy as np
import timeit
import pytest
from typing import Tuple
from agents.agent_minimax.minimax import score_board, generate_move_minimax
from game_utils import BoardPiece, SavedState, NO_PLAYER, PLAYER1, PLAYER2

#Fixed the names of test, all passed.
def test_score_board_empty():
    """
    Test that the empty board is assigned a score of 0.
    """
    from agents.agent_minimax.minimax import score_board
    # test on an empty board
    board_2_test = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    score = score_board(board_2_test, player, saved_state)
    assert score == 0  

def test_score_board_two_pieces():
    """
    Test that two pieces should be assigned score above 50.
    """
    from agents.agent_minimax.minimax import score_board
    board_2_test = np.array([[1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    score = score_board(board_2_test, player, saved_state)
    assert score > 50 

def test_score_board_opponent():
    """
    Test that 3 opponent pieces should be assigned negative scores.
    """
    from agents.agent_minimax.minimax import score_board
    board_2_test = np.array([[2, 2, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    score = score_board(board_2_test, player, saved_state)
    assert score < 0 


def test_score_window_win_minus_penalty():
    """
    Sample window that is evaluated, since three are connected a score above 100 should be assigned.
    """
    from agents.agent_minimax.minimax import score_board
    #four connected, three and one available, two and two available = all conditions
    board_2_test = np.array([[0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    score = score_board(board_2_test, player,saved_state)
    assert score > 100

# test_minimax_win(), test_generate_move_first(), generate_move_minimax(), test_generate_move_block()
# The function returns three values, but our test code only tried to unpack two of them, 
# resulting in the error ValueError: too many values to unpack (expected 2).

def test_minimax_win():
    """
    Test the scores given when there's a win.
    """
    from agents.agent_minimax.minimax import generate_move_minimax
    # four connected, three and one available, two and two available = all conditions
    board_2_test = np.array([[0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    move, _,_ = generate_move_minimax(board_2_test, player, saved_state) #fixed, failed to run since I changed the function to have more outputs
    assert move == 3
    depth = 4
    alpha = float('-inf')
    beta = float('inf')
    # Assigning function return values to move, saved_state, evaluated_moves variables
    move, saved_state, evaluated_moves = generate_move_minimax(board_2_test, player, saved_state)
    assert move == 3, "Expected the best move to complete a winning line"

# Remark: I'm not sure this test is super useful. But it's ok as a sanity check
def test_generate_move_first():
    """
    Test whether the move falls within the bounds of the board.
    """
    from agents.agent_minimax.minimax import generate_move_minimax
    board_2_test = np.array([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    best_move, _,_ = generate_move_minimax(board_2_test, player, saved_state)
    assert 0 <= best_move < 7 # Remark: I'm not sure this test is super useful. But it's ok as a sanity check
    best_move, _, _ = generate_move_minimax(board_2_test, player, saved_state)
    assert 0 <= best_move < 7, "The best move should be within the bounds of the board"
    
    
def test_generate_move_block():
    """
    Test if the minimax is blocking the opponent.
    """
    from agents.agent_minimax.minimax import generate_move_minimax
    board_2_test = np.array([[2, 2, 2, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    best_move, _, _ = generate_move_minimax(board_2_test, player, saved_state)
    assert best_move == 3, "Expected best move to block the opponent at column 3"

def test_generate_move_best():
    """
    the best move should be where 3 are already connected.
    """
    from agents.agent_minimax.minimax import generate_move_minimax
    board_2_test = np.array([[1, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    best_move, _,_ = generate_move_minimax(board_2_test, player, saved_state)
    assert best_move == 3


def test_generate_move_block():
    """
    Test generate_move_minimax to return the correct move for a simple winning condition using one liner.
    """
    # setting up a game state that satisfies simple victory conditions
    board = np.zeros((6, 7), dtype=int)
    # situation where PLAYER1 can win with the following moves:
    board[5, :] = [0, 0, 0, 1, 1, 1, 0]  
    player = BoardPiece(1)  
    saved_state = SavedState()
    best_move, _, _ = generate_move_minimax(board, player, saved_state)
    assert best_move == 4 or best_move == 0

def test_generate_move_minimax_evaluated_moves():
    """
    Test generate_move_minimax to evaluate at leaset one move.
    """
    # initial game board
    board = np.zeros((6, 7), dtype=int)  
    player = BoardPiece(1)
    saved_state = SavedState()
    
    _, _, evaluated_moves = generate_move_minimax(board, player, saved_state)
    assert evaluated_moves > 0
    

def initialize_game_state() -> np.ndarray:
    return np.zeros((6, 7), dtype=int)

# Function to drop a coin into a specific column
def apply_player_action(board: np.ndarray, action: int, player: int, copy: bool = True) -> np.ndarray:
    if copy:
        board = board.copy()
    for row in range(board.shape[0]-1, -1, -1):
        if board[row, action] == 0:
            board[row, action] = player
            break
    return board

# test cases to measure performance
def test_generate_minimax_performance():
    # prepare game board states that simulate the beginning, middle, and end states of the game.
    game_states = {
        'start': initialize_game_state(),
        'mid': initialize_game_state(),
        'end': initialize_game_state()
    }

    # mid-state example: Randomly drop a few coins.
    for _ in range(15):
        col = np.random.randint(0, 7)
        game_states['mid'] = apply_player_action(game_states['mid'], col, np.random.choice([1, 2]))

    # end-state example: Drop more coins, almost ending the game.
    for _ in range(35):
        col = np.random.randint(0, 7)
        game_states['end'] = apply_player_action(game_states['end'], col, np.random.choice([1, 2]))

    for state_name, board in game_states.items():
        player = 1  # select players for testing
        timer = timeit.Timer(lambda: generate_move_minimax(board, player, None))
        exec_time = timer.timeit(number=10)
        print(f"{state_name} state: {exec_time:.5f} seconds for 10 runs")

# run tests using pytest
if __name__ == "__main__":
    test_generate_minimax_performance()