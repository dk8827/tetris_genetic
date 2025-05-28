# piece.py
import random
from shapes import SHAPE_LOOKUP
from config import GRID_COLS

class Piece:
    def __init__(self, column, row, shape_idx):
        self.x = column
        self.y = row
        self.shape_idx = shape_idx
        self.color = SHAPE_LOOKUP.get_shape_color(shape_idx)
        self._num_rotations = SHAPE_LOOKUP.get_num_rotations(shape_idx)
        self.rotation = 0  # Rotation state index

    def get_relative_coords(self):
        """Returns the tuple of (dx, dy) relative coordinates for the current rotation."""
        return SHAPE_LOOKUP.get_shape_coords(self.shape_idx, self.rotation)

    def get_absolute_coords(self):
        """Returns a list of (absolute_x, absolute_y) grid coordinates for each block of the piece."""
        relative_coords = self.get_relative_coords()
        return [(self.x + dx, self.y + dy) for dx, dy in relative_coords]

    def rotate(self, direction=1):
        """Rotates the piece. Direction can be 1 (clockwise) or -1 (counter-clockwise)."""
        self.rotation = (self.rotation + direction) % self._num_rotations
        if self.rotation < 0: # Ensure positive index if direction was -1
            self.rotation += self._num_rotations


    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    @classmethod
    def get_new_piece(cls):
        """Creates a new random piece, typically starting at the top-middle of the board."""
        shape_idx = SHAPE_LOOKUP.get_random_shape_idx()
        # Start piece in middle-top. Y=0 is the first visible row.
        # Adjust Y based on piece's lowest point if pivot makes it spawn too low.
        # For now, y=0 is fine, as valid_space allows y < 0.
        return cls(column=GRID_COLS // 2, row=0, shape_idx=shape_idx)

    @classmethod
    def from_properties(cls, x, y, shape_idx, rotation):
        """Creates a piece with specific properties, useful for AI simulation."""
        piece = cls(x, y, shape_idx)
        piece.rotation = rotation
        return piece