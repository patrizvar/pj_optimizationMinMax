import numpy as np
import pytest
import pandas as pd
import time
from agents.agent_minimax.minimax import score_board, generate_move_minimax
from game_utils import BoardPiece, SavedState, NO_PLAYER, PLAYER1, PLAYER2

# 테스트 결과를 저장할 DataFrame 초기화
benchmarking_data = pd.DataFrame(columns=['Test Case', 'Execution Time (s)', 'Evaluated Moves', 'Accuracy'])

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

def test_generate_move_minimax_accuracy():
    """
    Test that generate_move_minimax returns the correct move for a simple winning condition.
    """
    # setting up a game state that satisfies simple victory conditions
    board = np.zeros((6, 7), dtype=int)
    # situation where PLAYER1 can win with the following moves:
    board[5, :] = [0, 0, 0, 1, 1, 1, 0]  
    player = BoardPiece(1)  
    saved_state = SavedState()
    
    best_move, _, _ = generate_move_minimax(board, player, saved_state)
    assert best_move == 3 or best_move == 6, "Expected best move to be 3 or 6 to win"

def test_generate_move_minimax_evaluated_moves():
    """
    Test that generate_move_minimax evaluates a reasonable number of moves.
    """
    # initial game board
    board = np.zeros((6, 7), dtype=int)  
    player = BoardPiece(1)
    saved_state = SavedState()
    
    _, _, evaluated_moves = generate_move_minimax(board, player, saved_state)
    assert evaluated_moves > 0, "Expected the algorithm to evaluate at least one move"

def record_test_result(test_name, expected_move, actual_move, execution_time, evaluated_moves, result):
    global benchmarking_results
    benchmarking_results = benchmarking_results.append({
        'Test Name': test_name,
        'Expected Move': expected_move,
        'Actual Move': actual_move,
        'Execution Time': execution_time,
        'Evaluated Moves': evaluated_moves,
        'Result': result
    }, ignore_index=True)

@pytest.mark.parametrize("board, player, expected_move", [
    # pLAYER1's first move on an empty board
    # expect to move to the center
    (np.zeros((6, 7), dtype=int), BoardPiece(1), 3), 
    # situation where PLAYER1 wins if he makes one more move
    # expect PLAYER1 to move to (5,2) and win
    (np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [2, 2, 0, 1, 0, 0, 0],
        [1, 1, 2, 1, 0, 0, 0],
        [1, 1, 1, 2, 2, 0, 0]
    ], dtype=int), BoardPiece(1), 2), 
    # defend a situation where PLAYER2 wins if he makes one more move
    # expect PLAYER1 to move to (5,2) and defend PLAYER2's victory
    (np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 2, 0, 0, 0],
        [2, 2, 1, 1, 0, 0, 0],
        [1, 2, 2, 1, 1, 0, 0]
    ], dtype=int), BoardPiece(1), 2),
])
def test_generate_move_minimax(board, player, expected_move):
    start_time = time.time()
    actual_move, _, evaluated_moves = generate_move_minimax(board, player, SavedState())
    execution_time = time.time() - start_time
    result = actual_move == expected_move
    test_name = f"Test {player} - Expected: {expected_move}, Actual: {actual_move}"
    record_test_result(test_name, expected_move, actual_move, execution_time, evaluated_moves, result)
    assert actual_move == expected_move, f"Expected move {expected_move}, but got {actual_move}"

@pytest.mark.parametrize("board, player, expected_move, expected_accuracy", [
    # player 1 is expected to select the center column as his first move
    (np.zeros((6, 7), dtype=int), PLAYER1, 3, True),
    # situation where player 1 reaches the victory condition
    (np.array([[0, 0, 0, PLAYER1, PLAYER1, PLAYER1, 0] + [0]*7*5]).reshape((6, 7)), PLAYER1, 3, True),
    # player 2 is close to the victory condition, player 1 must defend
    (np.array([[0, 0, 0, PLAYER2, PLAYER2, PLAYER2, 0] + [0]*7*5]).reshape((6, 7)), PLAYER1, 3, True),
])

def test_generate_move_minimax_and_benchmark(board, player, expected_move, expected_accuracy):
    saved_state = SavedState()
    start_time = time.time()
    actual_move, _, evaluated_moves = generate_move_minimax(board, player, saved_state)
    execution_time = time.time() - start_time
    accuracy = (actual_move == expected_move) == expected_accuracy

    # record results
    test_case_name = f"Test_{player}_Expected_{expected_move}_Actual_{actual_move}"
    benchmarking_data.append({
        'Test Case': test_case_name,
        'Execution Time (s)': execution_time,
        'Evaluated Moves': evaluated_moves,
        'Accuracy': accuracy
    }, ignore_index=True)

    assert actual_move == expected_move, f"Expected move {expected_move}, but got {actual_move}."
    assert accuracy is True, f"Accuracy expectation mismatch: expected {expected_accuracy}, got {accuracy}."

def save_benchmarking_results(filename="benchmarking_results.xlsx"):
    global benchmarking_results
    benchmarking_results.to_excel(filename, index=False)
    print(f"Benchmarking results saved to {filename}.")