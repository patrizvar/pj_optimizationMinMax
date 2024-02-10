import unittest
from agents.agent_bitboard.bitboard import Bitboard
from game_utils import PLAYER1, PLAYER2, GameState

class TestBitboard(unittest.TestCase):
    def test_initialization(self):
        bitboard = Bitboard()
        self.assertEqual(bitboard.width, 7)
        self.assertEqual(bitboard.height, 6)
        self.assertEqual(bitboard.current_position, 0)
        self.assertEqual(bitboard.mask, 0)
        self.assertEqual(bitboard.moves, 0)

    def test_can_play(self):
        bitboard = Bitboard()
        # Fill the column 0 completely.
        for row in range(bitboard.height):
            bitboard.mask |= 1 << (row + 0 * (bitboard.height + 1))

        # Now attempt to mark the top cell as filled.
        bitboard.mask |= 1 << (bitboard.height + 0 * (bitboard.height + 1))

        # Expect can_play to return False since the column is full.
        self.assertFalse(bitboard.can_play(0), "Should not be able to play in a full column")


    def test_play(self):
        bitboard = Bitboard()
        column = 0
        # Test playing a valid move.
        try:
            bitboard.play(0)
        except ValueError:
            self.fail("play(0) raised ValueError unexpectedly!")

        self.assertNotEqual(bitboard.mask, 0, "Mask should be updated after a valid move.")
        self.assertEqual(bitboard.moves, 1, "Moves should be incremented after playing.")

        # Fill the column to simulate a full column scenario.
        for _ in range(bitboard.height):
            bitboard.play(column)

        # Now, the column is full. Attempting to play in the same column should raise a ValueError.
        with self.assertRaises(ValueError, msg="Expected ValueError when trying to play in a full column"):
            bitboard.play(column)

    def test_column_height(self):
        bitboard = Bitboard()
        for i in range(bitboard.height):
            bitboard.play(0)
            self.assertEqual(bitboard.column_height(0), i + 1)

    def test_is_win(self):
        bitboard = Bitboard()
        # Test horizontal win
        for i in range(4):
            bitboard.play(i)  # Simulate Player 1 move
            if i < 3:  # Avoid extra moves beyond the win condition setup
                bitboard.play(i + bitboard.width)  # Simulate Player 2 move in different columns if necessary
        self.assertTrue(bitboard.is_win(), "Horizontal win condition not detected.")
        
        bitboard.reset()
        # Test vertical win
        for _ in range(4):
            bitboard.play(0)  # Player 1 move in column 0
            if _ < 3:  # To prevent an unnecessary extra move after setting up the win condition
                bitboard.play(1)  # Player 2 move in a different column to ensure alternation but not interfere with the win condition

        self.assertTrue(bitboard.is_win(), "Vertical win condition not detected.")

        bitboard.reset()
        # Test diagonal win (\ direction)
        for col in range(4):
            # Place pieces below the diagonal for proper setup.
            for _ in range(col):
                bitboard.play(col)  # Player 1
                bitboard.play(col)  # Player 2 to simulate filling and alternating turns.
            bitboard.play(col)  # Winning move for Player 1 in the diagonal.
        self.assertTrue(bitboard.is_win(), "Diagonal win condition not detected.")

        bitboard.reset()
        # Test diagonal win (/ direction)
        for i in range(3, -1, -1):
            for j in range(3 - i):
                bitboard.play(i)  # Simulate filling underneath the diagonal
                bitboard.play(i)  # Player 2 move to avoid immediate win by Player 1
            bitboard.play(i)  # Player 1 move to form the diagonal
        self.assertTrue(bitboard.is_win())

    def test_visualize_bitboard(self):
        bitboard = Bitboard()
        visualization = bitboard.visualize_bitboard()
        self.assertIsInstance(visualization, str)
        self.assertIn('0000000', visualization)  # Check for the expected output format

    def test_apply_move(self):
        bitboard = Bitboard()  # Correctly instantiate a Bitboard object for this test
        
        # Attempt to make a move in column 0. No need to specify PLAYER1 or PLAYER2.
        bitboard.play(0)
        self.assertEqual(bitboard.moves, 1, "Moves should be incremented after playing.")
        self.assertNotEqual(bitboard.mask, 0, "Mask should be updated after a valid move.")
        
        # The next play would be by the other player, given the alternation is handled internally.
        bitboard.play(1)  # Attempt to play in a different column to simulate a turn by the other player.
        self.assertEqual(bitboard.moves, 2, "Moves should be incremented after the second play.")

    def test_check_game_end(self):
        bitboard = Bitboard()
        # Assuming the bottom row is row 0, simulate a horizontal win in the first row.
        for col in range(4):
            bitboard.play(col)  # Player 1's move for the win.
            if col < 3:  # Ensure Player 2 plays in another column, not to interfere with Player 1's winning condition.
                bitboard.play(col + 4)  # Player 2's move in a column that doesn't affect the win condition.
        
        self.assertTrue(bitboard.is_win(), "Winning condition not detected correctly.")

        bitboard.reset()
        # # Simulate draw condition by filling the board without creating a win condition
        # # Specific setup for a draw...
        # is_draw = bitboard.moves == bitboard.width * bitboard.height and not bitboard.is_win()
        # self.assertTrue(bitboard.is_draw(), "Draw condition not detected correctly.")

        bitboard.reset()
        # Test for ongoing game by making a move and checking neither win nor draw conditions are met
        bitboard.play(0)
        ongoing_game = not bitboard.is_win() and bitboard.moves < bitboard.width * bitboard.height
        self.assertTrue(ongoing_game, "Ongoing game condition not detected correctly.")
        
    def test_pretty_print(self):
        bitboard = Bitboard()
        pretty = bitboard.pretty_print()
        self.assertIsInstance(pretty, str)
        self.assertIn('|0 1 2 3 4 5 6 |', pretty)  # Check for the expected output format

    def test_reset(self):
        bitboard = Bitboard()
        bitboard.play(0)
        bitboard.reset()
        self.assertEqual(bitboard.current_position, 0)
        self.assertEqual(bitboard.mask, 0)
        self.assertEqual(bitboard.moves, 0)

if __name__ == "__main__":
    unittest.main()