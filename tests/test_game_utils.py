import numpy as np
from game_utils import BoardPiece, NO_PLAYER

# Remark: Good job with the tests here! You should try to simplify the setup within the tests
# a little, but overall I'm happy with the tests. Another possible improvement would be to
# give the test functions more expressive names, i.e. use the names of the test functions
# to describe what the test does.

def test_initialize_game_state():
    """
    Test if the game state is initialized as expected.
    """
    from game_utils import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_pretty_print_board():
    """
    Test if the board is printed in the way expected. Board result (board_res)
    is manually created and then compared to the output of pretty_print_board, where 
    a numpy array of 0 was called.
    """
    from game_utils import pretty_print_board
    board_2_test = np.zeros((6, 7), dtype=int)
    board_res = (
        "|==============|\n" +
        "|              |\n" + 
        "|              |\n" +
        "|              |\n" +
        "|              |\n" +
        "|              |\n" +
        "|              |\n"
        "|==============|\n" +
        "|0 1 2 3 4 5 6 |\n")
    assert pretty_print_board(board_2_test) == board_res

def test_example_board():
    """
    Test if the numpy array containing numbers for player pieces on 
    the board translate to the X or 0 in the string if they
    are printed in the correct location = hence reverse order.
    """
    from game_utils import pretty_print_board
    board_2_test = np.array([[0, 2, 2, 1, 1, 0, 0],
                      [0, 2, 1, 2, 2, 0, 0],
                      [0, 0, 2, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    board_res = (
        "|==============|\n" +
        "|              |\n" +
        "|              |\n" +
        "|    X X       |\n" +
        "|    O X X     |\n" +
        "|  O X O O     |\n" +
        "|  O O X X     |\n" +
        "|==============|\n" +
        "|0 1 2 3 4 5 6 |\n"
    )
    assert pretty_print_board(board_2_test) == board_res

def test_string_to_board():
    """
    Test if the printes board as a string is converted to the numpy array correctly. The printed board
    is empty, therefore the numpy array should contain the same dimensions (6,7) and zeros.
    """
    from game_utils import string_to_board
    board_2_test = (
        "|==============|\n" +
        "|              |\n" + 
        "|              |\n" +
        "|              |\n" +
        "|              |\n" +
        "|              |\n" +
        "|              |\n" +
        "|==============|\n" +
        "|0 1 2 3 4 5 6 |\n")
    
    board_res = np.zeros((6, 7), dtype=int)

    assert np.array_equal(string_to_board(board_2_test), board_res)

#for arrays you cannot use == to test equality
def test_example_string_to_board():
    """
    Testing the transltaion the string (printed board) back to the numpy array.
    """
    from game_utils import string_to_board
    board_2_test = str(
        "|==============|\n" +
        "|              |\n" +
        "|              |\n" +
        "|    X X       |\n" +
        "|    O X X     |\n" +
        "|  O X O O     |\n" +
        "|  O O X X     |\n" +
        "|==============|\n" +
        "|0 1 2 3 4 5 6 |\n"
    )

    board_res = np.array([[0, 2, 2, 1, 1, 0, 0],
                      [0, 2, 1, 2, 2, 0, 0],
                      [0, 0, 2, 1, 1, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]], dtype=int)
    
    assert np.array_equal(string_to_board(board_2_test), board_res)


def test_apply_player1_action():
    """
    Test if the player action piece is applied to the correct column position.
    """
    from game_utils import apply_player_action
    rows = 6
    cols = 7
    initial_board = np.zeros((rows, cols), dtype=int)

    PlayerAction = np.int8  # the column to be played
    BoardPiece = np.int8
    PLAYER1 = BoardPiece(1) 
    player1_action = PlayerAction(3)
    player1_piece = PLAYER1
    new_board = apply_player_action(initial_board.copy(), player1_action, player1_piece)

    test_board = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)

    assert np.array_equal(new_board, test_board)

def test_apply_player2_action2():
    """
    Test two players in the same column to make sure they are on top of each other.
    """
    from game_utils import apply_player_action
    rows = 6
    cols = 7
    initial_board = np.zeros((rows, cols), dtype=int)

    PlayerAction = np.int8  # the column to be played
    BoardPiece = np.int8
    PLAYER1 = BoardPiece(1) 
    player1_action = PlayerAction(3)
    player1_piece = PLAYER1
    new_board = apply_player_action(initial_board.copy(), player1_action, player1_piece)
    PLAYER2 = BoardPiece(2) 
    player2_action = PlayerAction(3)
    player2_piece = PLAYER2
    new_board = apply_player_action(new_board.copy(), player2_action, player2_piece)

    test_board = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)
   
    assert np.array_equal(new_board, test_board)

def test_apply_player_action_fullBoard():
    """
    Check if when applying the player action, the statement if column is full works.
    """
    import pytest
    from game_utils import apply_player_action
    rows, cols = 6, 7
    board = np.ones((rows, cols), dtype=int)
    PlayerAction = np.int8
    player1_action = PlayerAction(3) 
    PLAYER1 = BoardPiece(1)
    player1_piece = PLAYER1 

    with pytest.raises(ValueError, match="Column is full."):
        apply_player_action(board.copy(), player1_action, player1_piece)

def test_connected_four_vertical():
    """
     Test if connected four vertically are detected.
    """
    from game_utils import connected_four
    # Test for 4 connected vertically
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0]], dtype=int)
    PLAYER1 = BoardPiece(1)
    assert connected_four(board, PLAYER1) is True

def test_connected_four_horizontal():
    """
    Test if connected four horizontally are detected.
    """
    from game_utils import connected_four
    # Test for 4 connected horizontally
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 0, 0]], dtype=int)
    PLAYER1 = BoardPiece(1)
    assert connected_four(board, PLAYER1) is True

def test_connected_four_diagonal1():
    """
    Check if connected four diagnoally are detected.
    """
    from game_utils import connected_four
    # Test for 4 connected diagonally
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 2, 1, 0],
                     [0, 0, 0, 0, 0, 0, 1]], dtype=int)
    PLAYER1 = BoardPiece(1)
    assert connected_four(board, PLAYER1) is True

def test_connected_four_diagonal2():
    """
    Check if connected four diagonally from a different direction are detected.
    """
    from game_utils import connected_four
    # Test for 4 connected diagonally
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 1, 1, 0, 0, 0],
                     [0, 1, 2, 0, 0, 0, 0],
                     [2, 2, 1, 0, 0, 0, 0]], dtype=int)
    PLAYER1 = BoardPiece(1)
    assert connected_four(board, PLAYER1) is True

def test_connected_four_none():
    """
    No wins check.
    """
    from game_utils import connected_four
    # Test for no 4 connected
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 2, 1, 2, 1, 2]], dtype=int)
    PLAYER1 = BoardPiece(1)
    assert connected_four(board, PLAYER1) is False

def test_check_end_state():
    """
    Check if the game state at win is detected.
    """
    from game_utils import check_end_state
    from game_utils import initialize_game_state
    from game_utils import GameState
    PLAYER1 = BoardPiece(1)
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 0, 0]], dtype=int)
    result = check_end_state(board, PLAYER1)
    assert result == GameState.IS_WIN

def test_check_end_state_draw():
    """
    Check if the game state at draw is detected.
    """
    from game_utils import check_end_state
    from game_utils import GameState
    PLAYER1 = BoardPiece(1)
    board = np.array([[1, 2, 1, 2, 1, 2, 1],
                     [2, 1, 2, 1, 1, 1, 2],
                     [4, 3, 4, 3, 4, 3, 4],
                     [2, 1, 2, 1, 2, 1, 2],
                     [4, 3, 4, 3, 4, 4, 3],
                     [1, 2, 3, 1, 2, 3, 4]], dtype=np.int8)
    result = check_end_state(board, PLAYER1)
    # Remark: It's a bit dangerous that you use pieces that are not actually used in the game. 
    # However, it's good you made sure that there is no connected four even for these pieces.
    assert result == GameState.IS_DRAW

def test_check_end_state_still_playing():
    """
    Check if when therea re still avaialble moves, the game state is still playing.
    """
    from game_utils import check_end_state
    from game_utils import GameState
    PLAYER1 = BoardPiece(1)
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 2, 1, 2, 0, 0]], dtype=np.int8)
    result = check_end_state(board, PLAYER1)
    assert result == GameState.STILL_PLAYING