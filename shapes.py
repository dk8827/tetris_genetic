# shapes.py
import random
from config import COLOR_GREEN, COLOR_RED, COLOR_CYAN, COLOR_YELLOW, COLOR_ORANGE, COLOR_BLUE, COLOR_MAGENTA

# Shape templates (original string format)
_S_SHAPE_TEMPLATE = [['.....',
                      '.....',
                      '..00.',
                      '.00..',
                      '.....'],
                     ['.....',
                      '..0..',
                      '..00.',
                      '...0.',
                      '.....']]
# ... (Z, I, O, J, L, T templates similarly) ...
_Z_SHAPE_TEMPLATE = [['.....', '.....', '.00..', '..00.', '.....'], ['.....', '..0..', '.00..', '.0...', '.....']]
_I_SHAPE_TEMPLATE = [['..0..', '..0..', '..0..', '..0..', '.....'], ['.....', '0000.', '.....', '.....', '.....']]
_O_SHAPE_TEMPLATE = [['.....', '.....', '.00..', '.00..', '.....']]
_J_SHAPE_TEMPLATE = [['.....', '.0...', '.000.', '.....', '.....'], ['.....', '..00.', '..0..', '..0..', '.....'], ['.....', '.....', '.000.', '...0.', '.....'], ['.....', '..0..', '..0..', '.00..', '.....']]
_L_SHAPE_TEMPLATE = [['.....', '...0.', '.000.', '.....', '.....'], ['.....', '..0..', '..0..', '..00.', '.....'], ['.....', '.....', '.000.', '.0...', '.....'], ['.....', '.00..', '..0..', '..0..', '.....']]
_T_SHAPE_TEMPLATE = [['.....', '..0..', '.000.', '.....', '.....'], ['.....', '..0..', '.00..', '..0..', '.....'], ['.....', '.....', '.000.', '..0..', '.....'], ['.....', '..0..', '..00.', '..0..', '.....']]


SHAPE_TEMPLATES_RAW = [_S_SHAPE_TEMPLATE, _Z_SHAPE_TEMPLATE, _I_SHAPE_TEMPLATE, _O_SHAPE_TEMPLATE,
                       _J_SHAPE_TEMPLATE, _L_SHAPE_TEMPLATE, _T_SHAPE_TEMPLATE]

SHAPE_COLORS = [COLOR_GREEN, COLOR_RED, COLOR_CYAN, COLOR_YELLOW,
                COLOR_ORANGE, COLOR_BLUE, COLOR_MAGENTA] # S, Z, I, O, J, L, T

# Pre-calculate relative coordinates for all shapes and rotations
# Each entry in PRECALCULATED_SHAPES is a tuple of rotations for one shape type.
# Each rotation is a tuple of (dx, dy) coordinates relative to the piece's anchor.
PRECALCULATED_SHAPES_COORDS = []
for shape_template_list in SHAPE_TEMPLATES_RAW:
    rotations_for_shape = []
    for shape_matrix_str_list in shape_template_list:
        coords = []
        for i, line_str in enumerate(shape_matrix_str_list):
            for j, char_in_line in enumerate(line_str):
                if char_in_line == '0':
                    # Offset to center piece around its x,y.
                    # Original logic: j-2 for x, i-4 for y.
                    # This means the pivot is roughly (2,4) in the 5x5 grid.
                    # For a 5x5, (2,2) is center. Let's adjust to a more standard pivot.
                    # If we consider (0,0) of the piece to be its logical rotation center.
                    # Let's adjust the pivot to be at (matrix col 2, matrix row 2)
                    # So, '0' at (2,2) in matrix -> (0,0) relative.
                    # '0' at (0,0) in matrix -> (-2,-2) relative.
                    coords.append((j - 2, i - 2)) # Centered around (2,2) of the 5x5 matrix
        rotations_for_shape.append(tuple(coords))
    PRECALCULATED_SHAPES_COORDS.append(tuple(rotations_for_shape))


class ShapeLookup:
    def __init__(self):
        self.shapes = PRECALCULATED_SHAPES_COORDS
        self.colors = SHAPE_COLORS
        self.raw_templates = SHAPE_TEMPLATES_RAW # For next shape preview if needed

    def get_random_shape_idx(self):
        return random.randrange(len(self.shapes))

    def get_shape_coords(self, shape_idx, rotation_idx):
        num_rotations = len(self.shapes[shape_idx])
        return self.shapes[shape_idx][rotation_idx % num_rotations]

    def get_shape_color(self, shape_idx):
        return self.colors[shape_idx]

    def get_num_rotations(self, shape_idx):
        return len(self.shapes[shape_idx])

    def get_raw_template(self, shape_idx, rotation_idx):
        num_rotations = len(self.raw_templates[shape_idx])
        return self.raw_templates[shape_idx][rotation_idx % num_rotations]

SHAPE_LOOKUP = ShapeLookup() # Global instance