from typing import Callable
from game_utils import GenMove
from agents.agent_human_user import user_move
from agents.agent_random import generate_move
from agents.agent_minimax import generate_minimax
import pandas as pd

# set to True during development or testing, set to False when deploying.
DEBUG_MODE = True

def timed_minimax(board, player, saved_state, args):
    import time

    start_time = time.time()
    evaluated_moves = 0

    def generate_move_wrapper(board, player, saved_state, *args):
        nonlocal evaluated_moves
        evaluated_moves += 1
        return generate_minimax(board, player, saved_state, *args)

    while time.time() - start_time < 2:
        generate_move_wrapper(board.copy(), player, saved_state, *args)

    return evaluated_moves

def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    import time
    from game_utils import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState, MoveStatus
    from game_utils import initialize_game_state, pretty_print_board, apply_player_action, check_end_state, check_move_status

    # measure time when in debug mode
    if DEBUG_MODE:
        game_start_time = time.time()
    
    players = (PLAYER1, PLAYER2)
    evaluated_moves_data = {player_1: []}

    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )

                if gen_move == generate_minimax:
                    action, saved_state, evaluated_moves = gen_move(
                        board.copy(),  # copy board to be safe, even though agents shouldn't modify it
                        player, saved_state, *args
                    )
                    evaluated_moves_data[player_name].append(evaluated_moves)
                    print(f'Moves evaluated in 2 seconds: {evaluated_moves}')

                else:
                    # Make sure to use the correct variable name here
                    action, saved_state = gen_move(
                        board.copy(),  # copy board to be safe, even though agents shouldn't modify it
                        player, saved_state, *args
                    )

                print(f'Move time: {time.time() - t0:.3f}s')

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    print(f'Move {action} is invalid: {move_status.value}')
                    print(f'{player_name} lost by making an illegal move.')
                    playing = False
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print('Game ended in draw')
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                    playing = False
                    break

    if DEBUG_MODE:
        game_end_time = time.time()
        print(f"Game completion time: {game_end_time - game_start_time}초")
    # Save evaluated moves data to Excel
    df = pd.DataFrame(evaluated_moves_data)
    df.to_excel('evaluated_moves_data.xlsx', index=False)

if __name__ == "__main__":
    #human_vs_agent(user_move)
    # human_vs_agent(generate_move)
    human_vs_agent(generate_minimax)
