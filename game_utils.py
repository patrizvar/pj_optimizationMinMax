from enum import Enum
import numpy as np
from typing import Callable, Optional

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input is not a number.'
    NOT_INTEGER = ('Input is not an integer, or isn\'t equal to an integer in '
                   'value.')
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

def initialize_game_state() -> np.ndarray:
    """
    Returns an np array, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """

    #np.full initializes an array with given values for shape (BOARD_SHAPE), data represenation (NO_PLAYER). 
    #dtype is BoardPiece, hence np.int8
    ret=np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)  # Remark: np.zeros() is a bit easier maybe
    return ret


def pretty_print_board(board: np.ndarray) -> str:
    """
    # Remark: Docstring description not updated
    # Remark: Is this a standard format?
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    Input parameters:
    -board (np.ndarray): the game board represented in the np.array format

    """
    #top of the board
    new_board="|==============|\n"

    # Remark: The block comment below is unnecessary
    #body of the board = loop over the board, rows in the reverse order, 
    #checks for the player1 and player2 and converts them to the string
    #rows = i; cols = j
    # Remark: You could iterate over the board array directly, instead of using indices
    for i in range(BOARD_ROWS-1,-1,-1):  # Remark: i and j are poor names, use more descriptive ones.
        new_board += "|"
        for j in range(BOARD_COLS):
            # Remark: Extract the block below for readability and testability
            if board[i, j] == 0:  # Remark: Compare to type BoardPiece, not to a bare integer
                new_board += " " + " "
            elif board[i, j] == 1:
                new_board += str(PLAYER1_PRINT) + " " 
            elif board[i, j] == 2:
                new_board += str(PLAYER2_PRINT) + " "  
        new_board += "|\n"

    #bottom of the board
    new_board += "|==============|\n"

    #numbers at the bottom of the board
    new_board += "|0 1 2 3 4 5 6 |\n"
    return new_board

def string_to_board(new_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.

    Input parameters:
    - new_board = represtend the updated board as a string which will be convereted to np.array

    """
    # Remark: too many comments
    #strip removes the spaces, split creates substrings divided based on \n occurenece
    get_rows = new_board.strip().split('\n')  # Remark: Not sure if strip is necessary

    #reiterate through strings in the reverse order, excludes '|', and updates the variable without it
    # i=row; j=line
    for i in range(len(get_rows)):
        get_rows[i] = get_rows[i][1:-2]
    get_rows=get_rows[1:-1]
    
    #loops through the get_rows which is string to convert it back to the numpy array
    #evaluates eachcolumn for X and 0 to assign it either to player1 or player2 respectively
    board = np.zeros((BOARD_ROWS, BOARD_COLS))  # Remark: make sure the dtype is BoardPiece
    for i, j in enumerate(get_rows):  # Remark: choose more descriptive variable names
        for col in range(BOARD_COLS):
            cell_str = j[col * 2] #to store the character from the string, *2 as there are gaps
            if cell_str == "X":  # Remark: Use type BoardPiecePrint
                board[BOARD_ROWS-1-i, col] = 1
            elif cell_str == "O":
                board[BOARD_ROWS-1-i, col] = 2

    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand). You can always only fill out the lowest part of the row in the
    column. So essentially, if row 1  and column 1 is taken, you move to row 2 if the column 1 is 
    selected again.

    Input parameters:
    -board: np.array representing the board
    -player: BoardPiece as integer, representing the player
    -action: PlayerAction as integer, representing which column is selected by the player

    Steps:
    - loops through the column from the lowest row working upwards
    - if there is an available row, player can put the piece there (which row it is is saved in 'i')
    - the modified board is returned
    - if the row is full, ValueError is raised

    """
    #lowest row in the column the player chooses; 
    #i = row, j = column
    # Remark: You can also use np.where(board[:, action] == NO_PLAYER)[0]
    for i in range(board.shape[0]): #iterates from the lowest row to the highest
        if board[i, action] == 0:  # Remark: Use type BoardPiece, not bare integer
            board[i, action] = player
            return board  # Remark: No need to return the modified board, it's modified in place
    
    #if the column is full, raisese a ValueError
    # Remark: No need to check for validity, this should happen outside this function
    raise ValueError("Column is full.")


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.

    Input parameters:
    -board: np.array representing the board
    -player: BoardPiece as integer, representing the player

    Steps:
    Loops through the array in 4 directions = vertical, horizontal, or diagonal (two-way). 
    Checks if thereare four pieces of any of the players (defined by the BoardPiece).
    
    """
    rows, cols = board.shape # i = row; j = column, n = number  # Remark: You can use BOARD_ROWS and BOARD_COLS 
    # Remark: It would be better to split this function up into smaller functions.
    # This would improve testability, and you could reuse the horizontal part to
    # check the columns (by applying it to the transposed board) and one diagonal part
    # to check the other diagonal (by applying it to the flipped board).
    # Note that you can't use BOARD_ROWS and BOARD_COLS anymore in that case.

    for i in range(rows): # Remark: You could iterate over the board directly
                          # Remark: i and j are poor names
                          # Remark: 3 and 4 are hard-coded values for the offset
                          #         from the board edge and the number of pieces needed for a win.
                          #         It would make your code more readable if you made them variables
                          #         with expressive names, which you could reuse 
        for j in range(cols):
            # horizontal
            if j + 3 < cols:
                if all(board[i, j + n] == player for n in range(4)):
                    return True
            # vertical
            if i + 3 < rows:
                if all(board[i + n, j] == player for n in range(4)):
                    return True
            # diagonal1
            if j + 3 < cols and i + 3 < rows:
                if all(board[i + n, j + n] == player for n in range(4)):
                    return True
            # diagonal2
            if j - 3 >= 0 and i + 3 < rows:
                if all(board[i + n, j - n] == player for n in range(4)):
                    return True

    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?

    Input parameters:
    - board: the current game board as a np.array
    - player: which player piece is used

    Steps:
    - two if statements, checkign if the current player won (GameState.IS_WIN), or whether no 
    more moves are available (the board is full, therefore GameState.IS_DRAW). If none of the conditions
    apply, the game is still going on

    """
    if connected_four(board, player):
        return GameState.IS_WIN

    if np.all(board != NO_PLAYER):
        return GameState.IS_DRAW

    else:
        return GameState.STILL_PLAYING


from typing import Callable, Optional

def check_move_status(board: np.ndarray, column: any) -> MoveStatus: #had to change from Any to any
    """
    Returns a MoveStatus indicating whether a move is legal or illegal, and why 
    the move is illegal.
    Any column type is accepted, but it needs to be convertible to a number
    and must result in a whole number.
    Furthermore, the column must be within the bounds of the board and the
    column must not be full.
    """
    try:
        numeric_column = float(column)
    except ValueError:
        return MoveStatus.WRONG_TYPE

    is_integer = np.mod(numeric_column, 1) == 0
    if not is_integer:
        return MoveStatus.NOT_INTEGER

    column = PlayerAction(column)
    is_in_range = PlayerAction(0) <= column <= PlayerAction(6)
    if not is_in_range:
        return MoveStatus.OUT_OF_BOUNDS

    is_open = board[-1, column] == NO_PLAYER
    if not is_open:
        return MoveStatus.FULL_COLUMN

    return MoveStatus.IS_VALID
