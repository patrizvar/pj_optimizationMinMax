from typing import Tuple
import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, PLAYER1, PLAYER2, connected_four

def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: SavedState) -> Tuple[PlayerAction, SavedState, int]:
    """
    Generate the best move.

    Input parameters:
    - board: np.array of the current board.
    - player (BoardPiece): Represents the player for whom the move is generated.
    - saved_state (SavedState): Represents the state of the game that might affect move generation.

    Steps:
    - Columns for each of the possible moves are considered.
    - Each of these moves is evaluated by the minimax.
    - Column index with the best move is returned.

    Returns:
    - Tuple[PlayerAction, SavedState, int]: Best move, game state, and number of evaluated moves are returned.
    """
    def minimax(board: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool, saved_state: SavedState, player: BoardPiece) -> float:
        """
        Applies the minimax algorithm to determine the best move for a player.

        Input parameters:
        - board: np.ndarray of the current board.
        - depth: int of the depth of the minimax search tree.
        - alpha: float for alpha pruning.
        - beta: float for beta pruning.
        - maximizing_player: bool to indicate if the player is maximizing.
        - saved_state: SavedState of the game. 
        - player: BoardPiece information about which player is making the move.

        Returns:
        - float: The best score achieved by the player.
        """
        if connected_four(board, player) or depth == 0:
            return score_board(board, player, saved_state)

        val = float('-inf') if maximizing_player else float('inf')
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                temp_board = board.copy()
                row = np.where(temp_board[:, col] == 0)[0][-1]
                temp_board[row, col] = player

                score = minimax(temp_board, depth - 1, alpha, beta, not maximizing_player, saved_state, player)

                if maximizing_player:
                    val = max(val, score)
                    alpha = max(alpha, val)
                else:
                    val = min(val, score)
                    beta = min(beta, val)

                if beta <= alpha:
                    break

        return val

    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    evaluated_moves = 0

    for col in range(board.shape[1]):
        if 0 in board[:, col]:
            evaluated_moves += 1
            temp_board = board.copy()
            row = np.where(temp_board[:, col] == 0)[0][-1]
            temp_board[row, col] = player

            depth = 4
            maximizing_player = True
            score = minimax(temp_board, depth, alpha, beta, maximizing_player, saved_state, player)

            if score > best_score:
                best_score = score
                best_move = col
                equal_moves = []
            elif score == best_score:
                equal_moves.append(col)

    if equal_moves:
        best_move = np.random.choice(equal_moves)

    return best_move, saved_state, evaluated_moves

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
        
        points_for_three = 100
        points_for_two = 50
        points_for_one = 1
        # Remark: Why doesn't the opponent get points for 4 or 2? Heuristics are usually symmetric.

        # add the scores based on the patterns detected
        if player_count == 3:
            score += points_for_three
        elif player_count == 2:
            score += points_for_two
        elif player_count == 1:
            score += points_for_one

        if opp_count == 3: 
            score -= points_for_three*6  # penalize opponent's win
        elif opp_count == 2:
            score -= points_for_two*4
        elif opp_count == 1:
            score -= points_for_one

        return score