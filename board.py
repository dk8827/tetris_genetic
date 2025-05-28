# board.py
from config import GRID_COLS, GRID_ROWS, COLOR_BLACK

class Board:
    def __init__(self):
        self.locked_positions = {}  # (x, y) -> color
        self.grid_width = GRID_COLS
        self.grid_height = GRID_ROWS

    def create_visual_grid(self):
        """Creates a 2D list representing the grid with colors for drawing."""
        grid = [[COLOR_BLACK for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        for r_idx in range(self.grid_height):
            for c_idx in range(self.grid_width):
                if (c_idx, r_idx) in self.locked_positions:
                    color = self.locked_positions[(c_idx, r_idx)]
                    grid[r_idx][c_idx] = color
        return grid

    def is_valid_space(self, piece):
        """Checks if the piece's current position is valid on the board."""
        abs_coords = piece.get_absolute_coords()
        for x_coord, y_coord in abs_coords:
            # Check horizontal boundaries
            if not (0 <= x_coord < self.grid_width):
                return False
            # Check bottom boundary (pieces can be above the top, y < 0)
            if y_coord >= self.grid_height:
                return False
            # Check collision with locked pieces (only if on the visible grid)
            if y_coord >= 0 and (x_coord, y_coord) in self.locked_positions:
                return False
        return True

    def lock_piece(self, piece):
        """Locks the piece onto the board."""
        for x_coord, y_coord in piece.get_absolute_coords():
            self.locked_positions[(x_coord, y_coord)] = piece.color

    def clear_rows(self):
        """
        Clears any full rows from the board.
        Returns the number of rows cleared.
        """
        increment = 0
        rows_to_clear = []

        # Iterate from bottom up to find full rows
        for r_idx in range(self.grid_height - 1, -1, -1):
            row_is_full = True
            for c_idx in range(self.grid_width):
                if (c_idx, r_idx) not in self.locked_positions:
                    row_is_full = False
                    break
            if row_is_full:
                increment += 1
                rows_to_clear.append(r_idx)
                # Remove blocks of this row from locked_positions
                for c_idx_del in range(self.grid_width):
                    if (c_idx_del, r_idx) in self.locked_positions:
                        del self.locked_positions[(c_idx_del, r_idx)]

        if increment > 0:
            new_locked = {}
            # Sort remaining locked positions by y-value (bottom-up)
            # to correctly shift them down.
            sorted_locked_keys = sorted(self.locked_positions.keys(), key=lambda k: k[1], reverse=True)

            for x_lock, y_lock in sorted_locked_keys:
                num_cleared_below_this_block = sum(1 for r_cleared in rows_to_clear if y_lock < r_cleared)
                new_y = y_lock + num_cleared_below_this_block # Add because rows are cleared below, blocks fall
                new_locked[(x_lock, new_y)] = self.locked_positions[(x_lock, y_lock)]
            
            self.locked_positions = new_locked
        return increment

    def check_lost(self):
        """Checks if any locked piece is above the visible play area (y < 0)."""
        for _, y_pos in self.locked_positions.keys():
            if y_pos < 0: # Top-most visible row is 0. Anything < 0 means lost.
                          # If pieces lock partially above (e.g. y=-1, y=0, y=1), this counts as loss.
                          # More standard Tetris loss: piece locks and any part is in row 0 (or row 1 if row 0 is buffer)
                          # and it cannot move down.
                          # The original check_lost(positions) looked for y < 1.
                          # Let's stick to: if any part of a locked piece is above row 0 (y<0)
                          # OR if a piece locks and its highest block is in row 0.
                          # Simpler: if any locked block is in y=0 (or above) and it means they couldn't spawn.
                          # The current logic: piece spawns at y=0. If it locks there immediately, check_lost applies.
                          # Original was `if y < 1`. So if a piece is in row 0 (y=0), it's a loss.
                return True
        return False

    def get_column_heights(self):
        """Returns a list of heights for each column (0 if empty)."""
        heights = [0] * self.grid_width
        visual_grid = self.create_visual_grid() # Use current state of locked_positions
        for c in range(self.grid_width):
            for r in range(self.grid_height):
                if visual_grid[r][c] != COLOR_BLACK:
                    heights[c] = self.grid_height - r
                    break
        return heights

    def get_aggregate_height(self):
        return sum(self.get_column_heights())

    def get_num_holes(self):
        holes = 0
        visual_grid = self.create_visual_grid()
        for c in range(self.grid_width):
            col_has_block_above = False
            for r in range(self.grid_height): # Top to bottom
                if visual_grid[r][c] != COLOR_BLACK:
                    col_has_block_above = True
                elif col_has_block_above and visual_grid[r][c] == COLOR_BLACK:
                    holes += 1
        return holes

    def get_bumpiness(self):
        heights = self.get_column_heights()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i+1])
        return bumpiness
    
    def clone(self):
        """Creates a deep copy of the board state."""
        new_board = Board()
        new_board.locked_positions = self.locked_positions.copy()
        return new_board