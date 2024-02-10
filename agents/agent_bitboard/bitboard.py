from game_utils import PLAYER1, PLAYER2, GameState

class Bitboard:
    def __init__(self, width=7, height=6):
        """Initialize the board with given width and height.
        current_position tracks the current player's stones on the board.
        mask tracks all stones on the board (both current player and opponent).
        moves count the number of moves made in the game."""
        self.width = width
        self.height = height
        self.current_position = 0  # current player's stone location
        self.mask = 0  # total stone positions (current player + opponent player)
        self.moves = 0  # number of moves performed

    def can_play(self, col):
        """Check if a stone can be placed in the given column.
        It verifies if the top cell of the column is empty."""
        top_cell = 1 << (col * (self.height + 1) + self.height)
        return (self.mask & top_cell) == 0

    def play(self, col):
        """Performs the action of placing a stone and updates the game state.
        It calculates the position to place the stone based on the column height,
        updates the current position and the mask."""
        if not self.can_play(col):
            raise ValueError("Column is full.")
        move = 1 << (self.column_height(col) + col * (self.height + 1))
        self.mask |= move
        if self.moves % 2 == 0:  # PLAYER1's turn
            self.current_position |= move
        self.moves += 1

    def column_height(self, col):
        """Calculates the height of stones in a given column,
        which is essential for determining where the next stone will land."""
        for row in range(self.height):
            if (self.mask & (1 << (col * (self.height + 1) + row))) == 0:
                return row
        return self.height

    def is_win(self):
        """Check if the current player has won.
        It calls check_alignment to verify if there are 4 consecutive stones aligned."""
        directions = [1, self.height + 1, self.height + 2, self.height]
        for direction in directions:
            m = self.current_position & (self.current_position >> direction)
            if m & (m >> (2 * direction)):
                return True
        return False
    
    def visualize_bitboard(self):
        visualization = "Bitboard Visualization:\n0000000\n"  # Extra top row for bottom
        for row in range(self.height - 1, -1, -1):
            line = ""
            for col in range(self.width):
                position = 1 << (row + col * (self.height + 1))
                if self.mask & position:
                    line += "1" if self.current_position & position else "0"
                else:
                    line += "0"
            visualization += line + "\n"
        visualization += "1111111"  # Bottom row as per the example
        return visualization
    
    def reset(self):
        self.current_position = 0
        self.mask = 0
        self.moves = 0

    def pretty_print(self):
        visualization = ""
        for row in range(self.height-1, -1, -1):
            visualization += "|"
            for col in range(self.width):
                pos = 1 << (row + col * (self.height + 1))
                if self.mask & pos == 0:
                    visualization += " O"
                elif self.current_position & pos == 0:
                    visualization += " O"
                else:
                    visualization += " X"
            visualization += " |\n"
        visualization += "|==============|\n"
        visualization += "|0 1 2 3 4 5 6 |\n"
        return visualization
    
    def check_game_end(self):
        if self.is_win():
            return GameState.IS_WIN
        elif self.moves == self.width * self.height:
            return GameState.IS_DRAW
        else:
            return GameState.STILL_PLAYING

    def is_draw(self):
        # Check if all positions are filled and there is no win
        return self.moves == self.width * self.height and not self.is_win()

def apply_move(self, col, player):
    if not self.can_play(col):
        raise ValueError("Column is full or invalid.")
    move = self.bottom_mask(col) << self.column_height(col)
    if player == PLAYER1:
        self.current_position ^= self.mask
    self.mask |= move
    self.moves += 1