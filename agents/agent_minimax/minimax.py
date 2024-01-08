from typing import Tuple
import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, PLAYER1, PLAYER2, connected_four

# Remark:
# Overall, this is a solid submission without major issues. Well done! 
# Positives:
# + Overall good tests
# + Overall good code style and readability
# + Overall solid documentation
# + Alpha-beta pruning correctly implemented
# + Reasonable heuristic
# Weaknesses:
# - Mistakes in minimax implementation, poor play as a result
# - Naming of variables and functions can be better
# - Tests in test_minimax.py don't work because the names of the functions don't exist in minimax.py
# - Too many comments
# - Non-standard docstring format?
# - Some refactoring steps could be done to make the code more readable

def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: SavedState) -> Tuple[PlayerAction,SavedState]:
    """
    Generate the best move.

    Input parameters:
    - board: np.array of the  current board.
    - player (BoardPiece): Represents the player for whom the move is generated.
    - saved_state (SavedState): Represents the state of the game that might affect move generation.

    Steps:
    - columns for each of the possible moves are considered
    - each of these moves is evaluated by the minimax
    - column index with the best move is returned

    Returns:
    - Tuple[PlayerAction, SavedState]: best move and game state is returned
    """
    # Remark:  I don't like that you separated the first loop over moves from the rest. I do get that it's a little easier to get
    #  the best move like that, but it leads to a lot of duplicated code and the logic is just the same for every depth.
    #  So there's no need to have two functions doing basically the same thing. Furthermore, this adds another level to
    #  the search depth and obscures how deep you're actually searching, and you’re losing some of the power of alpha-beta pruning.
    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    for col in range(board.shape[1]):
        if 0 in board[:, col]:  # check if there's a space in the column  # Remark: it would be more elegant to get the valid columns beforehand and only loop over those
            temp_board = board.copy()
            row = np.where(temp_board[:, col] == 0)[0][-1]
            temp_board[row, col] = player  # Remark: Why not use the apply_player_action function?
            score = minimax_function(temp_board, depth=4, alpha=alpha, beta=beta, maximizing_player=True, saved_state=saved_state, player=player)  # Remark: don't call the function using keyword arguments if you haven't specified them as such
            if score > best_score:
                best_score = score
                best_move = col
            #print(f"Score for column {col}: {score}")

    return best_move, saved_state

# minimax function
def minimax_function(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool, saved_state:SavedState, player: BoardPiece) -> float:
    """ 
    Applies the minimax algorithm to determine the best move for a player. Calls function score_board to assigns 
    scores and it also calls itself to look at more possible moves.

    Input parameters:
    - board: np.ndarray of the current board.
    - depth: int of the depth of the minimax search tree.
    - alpha: float for alpha pruning.
    - beta: float for beta pruning.
    - maximizing_player: bool to indicate if the player is maximizing.
    - saved_state: SavedState of the game. 
    - player: BoardPiece information about which player is making the move.

    Steps:
    - Check if the current player can win in the current state.
    - if maximizing player is playing:
        - go through possible moves and assign them scores using minimax
        - update alpha-beta values and returns the maximum scores that could be obtained.
    - if minimizing player is playing:
        - evaluates moves for the oponent using minimax
        - updates alpha-beta values and returns the minimum evaluated score.

    Returns:
    - float: The best score achieved by the player """
    # Remark: player and maximizing_player are a little confusing. Overall, I think fixing player 1 to be maximizing and
    #  player 2 to be minimizing is probably simpler. Or renaming maximizing_player to sth like 'is_root_player', which
    #  simply tells you whether the current player is the one for which you are looking for a move or not. You're actually
    #  doing this by always keeping the player variable the same and only making moves as the opponent if you're on the minimizing branch.
    #  You could therefore also rename 'player' to 'root_player', for example. 
    
    rows, columns = board.shape
    # check if there's a win or depth of the search tree is reached
    # Remark: There are multiple places where you check for game states. This is inefficient and should all be in one place.
    if connected_four(board, player) or depth == 0:  # Remark: what if the opponent made the last move?
        return score_board(board, player, saved_state)

    if maximizing_player:
        max_val = float('-inf')
        for col in range(columns):
            if 0 in board[:, col]:
                temp_board = board.copy()
                row = np.where(temp_board[:, col] == 0)[0][-1]
                temp_board[row, col] = player  # maximizing player's move

                # Check for immediate win  # Remark: this is a little redundant, since you already check for a win at the beginning of the function and you evaluate the board in minimax_function again
                if connected_four(temp_board, player):
                    return score_board(temp_board, player, saved_state) # Remark: If you find a win, you don't need to score anymore.

                val = minimax_function(temp_board, depth - 1, alpha, beta, False, saved_state, player)
                max_val = max(max_val, val)
                alpha = max(alpha, val) #alpha = max value found by max player
                if beta <= alpha:
                    break
        return max_val
    else:
        min_val = float('inf')
        for col in range(columns):
            if 0 in board[:, col]:
                temp_board = board.copy()
                row = np.where(temp_board[:, col] == 0)[0][-1]
                temp_board[row, col] = PLAYER1 if player == PLAYER2 else PLAYER2  # minimizing player's move  # Remark: Make this a function, you need that in several places

                # checking if the opponent's move leads to an immediate win
                # Remark: But you applied the move for the opponent, not for player, so you don't find the win for opponent here
                if connected_four(temp_board,player):
                    return score_board(temp_board, player, saved_state)
                
                val = minimax_function(temp_board, depth - 1, alpha, beta, False, saved_state, player)  # Remark: You need to call minimax with maximizing_player=True here
                min_val = min(min_val, val)
                beta = min(beta, val) # beta = min value found by min player
                if beta <= alpha: 
                    break
        return min_val

def score_board(board: np.ndarray, player: BoardPiece, saved_state: SavedState) -> int:
    # Remark: You're evaluating everything from the perspective of the player
    #  making the move at the root board.
    #  Another possibility is to evaluate everything from a fixed perspective
    #  of PLAYER1 and PLAYER2 (i.e., PLAYER1 always tries to maximize the 
    #  score and evaluations in PLAYER1s favor are positive, and vice versa).
    #  You wouldn't need the maximizing_player parameter then.
    # Another advantage would be that you wouldn't need to make the heuristic player-dependent.
    """
    Function used to calculate scores for a set player determined by BoardPiece.

    Input parameters:
    - board: np.array of the current board
    - player: BoardPiece determining for which player we are calculating the scores.
    - saved_state: SavedState of the board which is used for the evaluation.

    Steps:
    - iterates through all the rows and columns in horizontal, vvertical and diagonal fashion
    - scores each window of 4 pieces based on its composition by counting the number of pieces for the minimax player,
    human player and empty pieces in each window. Uses evlauate_window to assign the points.
    
    Return:
    - overall score across the board
    - int: the total score achieved for the player based on the board state """
    
    rows, columns = board.shape
    player_score = 0
    # Remark: See remarks in connected_four. Furthermore, there is potential for reusing code from that function here.
    # checking columns
    for col in range(columns):
        for row in range(rows - 3):
            window = board[row:row + 4, col]
            player_score += assign_scores(window, player)
    # checking rows
    for row in range(rows):
        for col in range(columns - 3):
            window = board[row, col:col + 4]
            player_score += assign_scores(window, player)
    # checking diagonals
    for row in range(rows - 3):
        for col in range(columns - 3):
            window = [board[row + i][col + i] for i in range(4)]
            player_score += assign_scores(window, player)
    # checking diagonals 2
    for row in range(rows - 3):
        for col in range(columns - 3):
            window = [board[row + 3 - i][col + i] for i in range(4)]
            player_score += assign_scores(window, player)

    return player_score

def assign_scores(window: np.ndarray, player:BoardPiece):
        """
    Evaluates a window of 4 pieces to calculate the score for the player.

    Input parameters:
    - window: np.array of a segment from the current board (row, column, diagonal)
    - player: determined by the BoardPiece, score of the player that is being calculated

    Steps:
    - checks the window to count the player's pieces, opponent's pieces,
       and empty slots
    - assigns scores based on the patterns (4 in the row, 3 in the and one empty slot, ...)
    - assigns penalty for the human player potential wins
    - the score achieved in the specific window is returned

    Returns:
    - int: the score obtained by the minimax player in the window that is being evaluated """

        # chceck which player is playing
        score = 0
        # Remark: Just setting opp_player = PLAYER1 if player == PLAYER2 else PLAYER2 would be enough, like you did above. No need to set player again.
        if player == BoardPiece(1):
            player = PLAYER1
            opp_player=PLAYER2
        elif player == BoardPiece(2):
            player = PLAYER2
            opp_player=PLAYER1
    
        # count the pieces for each player
        player_count = np.count_nonzero(window == player)
        opp_count = np.count_nonzero(window == opp_player)
        empty_count = np.count_nonzero(window == NO_PLAYER)

        points_for_four = 200
        points_for_three = 10
        points_for_two = 4
        #penalize the other player more, in order to block opponent from winnig
        penalty_for_other_player = 200
        # Remark: Why doesn't the opponent get points for 4 or 2? Heuristics are usually symmetric.

        # add the scores basedo n the patterns detected
        if player_count == 4:  # Remark: This is basically a win, which you have checked in minimax several times
            score += points_for_four
        elif player_count == 3 and empty_count == 1:
            score += points_for_three
        elif player_count == 2 and empty_count == 2:
            score += points_for_two
        if opp_count == 3 and empty_count == 1: 
            score -= penalty_for_other_player  # penalize opponent's win
        return score



