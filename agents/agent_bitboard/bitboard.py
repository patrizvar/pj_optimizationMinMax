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
        top_mask = self.top_mask(col)
        return (self.mask & top_mask) == 0

    def play(self, col):
        """Performs the action of placing a stone and updates the game state.
        It calculates the position to place the stone based on the column height,
        updates the current position and the mask."""
        move = self.bottom_mask(col) << self.column_height(col)
        self.current_position ^= self.mask
        self.mask |= move
        self.moves += 1

    def column_height(self, col):
        """Calculates the height of stones in a given column,
        which is essential for determining where the next stone will land."""
        return ((self.mask >> (col * (self.height + 1))) & ((1 << self.height) - 1)).bit_length()

    def is_win(self):
        """Check if the current player has won.
        It calls check_alignment to verify if there are 4 consecutive stones aligned."""
        return self.check_alignment(self.current_position)

    def check_alignment(self, position):
        """Check for victory conditions by verifying if there are 4 consecutive stones aligned
        in any direction (horizontal, vertical, diagonal)."""
        for direction in [1, self.height + 1, self.height, self.height + 2]:
            m = position & (position >> direction)
            if m & (m >> (2 * direction)):
                return True
        return False

    def top_mask(self, col):
        """Calculate the mask for the top cell of a column, used to check if the column is full."""
        return 1 << (self.height - 1 + col * (self.height + 1))

    def bottom_mask(self, col):
        """Calculate the mask for the bottom cell of a column, used as the starting point for a new stone."""
        return 1 << (col * (self.height + 1))
    
    def visualize_bitboard(self):
        visualization = "Bitboard Visualization:\n"
        for row in range(self.height - 1, -1, -1):  # 게임 보드의 실제 높이에 맞게 조정
            for col in range(self.width):
                position = 1 << (row + col * self.height)  # 수정된 비트 위치 계산
                if self.mask & position:
                    visualization += "1"
                else:
                    visualization += "0"
            visualization += "\n"
        return visualization

