import numpy as np
import timeit
import pytest
from typing import Tuple
from agents.agent_minimax.minimax import score_board, generate_move_minimax
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
    
# 게임 보드를 초기화하는 함수 (예시)
def initialize_game_state() -> np.ndarray:
    return np.zeros((6, 7), dtype=int)

# @pytest.mark.parametrize("board, player, expected_move", [
#     # 플레이어 1의 첫 번째 수는 중앙 열로 가정
#     (initialize_game_state(), 1, 3),
#     # 플레이어 1이 다음 수로 승리 가능
#     (np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [2, 2, 0, 1, 0, 0, 0],
#         [1, 1, 2, 1, 0, 0, 0],
#         [1, 1, 1, 2, 2, 0, 0]
#     ], dtype=int), 1, 2),
#     # 플레이어 1이 상대의 승리를 방어해야 함
#     (np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 2, 0, 0, 0],
#         [2, 2, 1, 1, 0, 0, 0],
#         [1, 2, 2, 1, 1, 0, 0]
#     ], dtype=int), 1, 2),
# ])
# def test_generate_minimax(board, player, expected_move):
#     actual_move = generate_move_minimax(board, player, SavedState())
#     assert actual_move == expected_move, f"Expected move {expected_move}, but got {actual_move}"




# 특정 열에 동전을 떨어뜨리는 함수 (예시)
def apply_player_action(board: np.ndarray, action: int, player: int, copy: bool = True) -> np.ndarray:
    if copy:
        board = board.copy()
    for row in range(board.shape[0]-1, -1, -1):
        if board[row, action] == 0:
            board[row, action] = player
            break
    return board

# 성능 측정을 위한 테스트 케이스
def test_generate_minimax_performance():
    # 게임의 시작, 중반, 종반 상태를 시뮬레이션하는 게임 보드 상태를 준비합니다.
    game_states = {
        'start': initialize_game_state(),
        'mid': initialize_game_state(),
        'end': initialize_game_state()
    }

    # 중반 상태 예시: 몇 개의 동전을 무작위로 떨어뜨립니다.
    for _ in range(15):
        col = np.random.randint(0, 7)
        game_states['mid'] = apply_player_action(game_states['mid'], col, np.random.choice([1, 2]))

    # 종반 상태 예시: 더 많은 동전을 떨어뜨려 거의 게임이 끝나게 합니다.
    for _ in range(35):
        col = np.random.randint(0, 7)
        game_states['end'] = apply_player_action(game_states['end'], col, np.random.choice([1, 2]))

    for state_name, board in game_states.items():
        player = 1  # 테스트를 위한 플레이어 선택
        timer = timeit.Timer(lambda: generate_move_minimax(board, player, None))
        exec_time = timer.timeit(number=10)
        print(f"{state_name} state: {exec_time:.5f} seconds for 10 runs")

# pytest를 사용하여 테스트 실행
if __name__ == "__main__":
    test_generate_minimax_performance()