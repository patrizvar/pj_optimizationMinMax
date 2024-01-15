import numpy as np
from agents.agent_minimax.minimax import score_board  
from game_utils import BoardPiece, SavedState, NO_PLAYER, PLAYER1, PLAYER2


# Remark: Good tests in principle, but the names are wrong so the tests don't work. 
# You probably changed something in the code and forgot to update the tests.
# However, I like how you tested the agent's behaviour in different situations.
# You could extend that to more complicated forced combinations of moves.

# Remark: The tests here don't work because the names of the functions don't exist in minimax*
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
    Test that two pieces should be assigned score 4.
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
    assert score == 57 

def test_score_board_opponent():
    """
    Test that 3 opponent pieces should be assigned scores of -200.
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
    assert score == -807 


def test_score_window_win_minus_penalty():
    """
    Sample window that is evaluated, since three are connected, an output of 10 is expected.
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
    assert score == 264

def test_minimax_win():
    """
    Test the scores given when there's a win.
    """
    from agents.agent_minimax.minimax import generate_move_minimax
    #four connected, three and one available, two and two available = all conditions
    board_2_test = np.array([[0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    depth = 4
    alpha = float('-inf')
    beta = float('inf')
    move, saved_state = generate_move_minimax(board_2_test, player, saved_state)
    assert move == 3

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
    best_move, new_saved_state = generate_move_minimax(board_2_test, player, saved_state)
    assert 0 <= best_move < 7 # Remark: I'm not sure this test is super useful. But it's ok as a sanity check

def test_generate_move_best():
    """
    the best move should be where 3 are already connected.
    """
    from agents.agent_minimax.minimax import generate_move_minimax
    board_2_test = np.array([[1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    player = BoardPiece(1)
    saved_state = SavedState()
    best_move, new_saved_state = generate_move_minimax(board_2_test, player, saved_state)
    assert best_move == 3

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
    best_move, new_saved_state = generate_move_minimax(board_2_test, player, saved_state)
    assert best_move == 3


