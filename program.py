import pygame
import random
import neat
import neat.graphs # Added for NN visualization
import os
import pickle # To save and load the best genome

# --- Pygame Setup ---
pygame.font.init()
pygame.mixer.init() # For sound effects (optional)

# Screen dimensions
S_WIDTH = 1200
S_HEIGHT = 700
PLAY_WIDTH = 300  # 300 // 10 = 30 width per block
PLAY_HEIGHT = 600 # 600 // 20 = 30 height per block
BLOCK_SIZE = 30

TOP_LEFT_X = (S_WIDTH - PLAY_WIDTH) // 2
TOP_LEFT_Y = S_HEIGHT - PLAY_HEIGHT - 50 # 50px padding at the bottom

# --- Shapes ---
# Shape formats:
# Each list represents a rotation of the shape.
# Each inner list is a row, and '0' represents a block.
S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

# Global variables for tracking the best genome for visualization
BEST_GENOME_EVER_FOR_VIZ = None
HIGHEST_FITNESS_EVER_FOR_VIZ = -float('inf')

# --- Piece Class ---
class Piece:
    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0

# --- Game Logic Functions ---
def create_grid(locked_positions={}):
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)] # 10 cols, 20 rows
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if (c,r) in locked_positions:
                color = locked_positions[(c,r)]
                grid[r][c] = color
    return grid

def convert_shape_format(piece):
    positions = []
    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((piece.x + j, piece.y + i))

    # Offset positions to be relative to top-left of the format
    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4) # Adjust based on 5x5 shape format padding
    return positions

def valid_space(piece, grid):
    accepted_positions = [[(j, i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub] # Flatten list

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
        if pos not in accepted_positions:
            if pos[1] > -1: # Allow pieces to be above the screen initially
                return False
    return True

def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False

def get_shape():
    return Piece(5, 0, random.choice(shapes)) # Start piece in middle-top

def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, 1, color)
    surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH/2 - (label.get_width()/2),
                         TOP_LEFT_Y + PLAY_HEIGHT/2 - label.get_height()/2 - 150)) # Adjusted Y

def draw_grid_lines(surface, grid):
    sx = TOP_LEFT_X
    sy = TOP_LEFT_Y
    for i in range(len(grid)): # Rows
        pygame.draw.line(surface, (128,128,128), (sx, sy + i*BLOCK_SIZE), (sx + PLAY_WIDTH, sy + i*BLOCK_SIZE))
    for j in range(len(grid[0])): # Columns
        pygame.draw.line(surface, (128,128,128), (sx + j*BLOCK_SIZE, sy), (sx + j*BLOCK_SIZE, sy + PLAY_HEIGHT))

def clear_rows(grid, locked):
    increment = 0 # Number of rows cleared
    for i in range(len(grid)-1, -1, -1): # Iterate from bottom up
        row = grid[i]
        if (0,0,0) not in row: # If row is full
            increment += 1
            ind = i # Row index to remove
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue
    # Shift rows down
    if increment > 0:
        # Sort locked positions by y-value (row) in descending order
        sorted_locked = sorted(list(locked.keys()), key=lambda x: x[1], reverse=True)
        for x, y in sorted_locked:
            if y < ind: # Only shift rows above the cleared row(s)
                new_key = (x, y + increment) # Shift down by number of cleared rows
                locked[new_key] = locked.pop((x,y))
    return increment


def draw_next_shape(piece, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255,255,255))

    sx = TOP_LEFT_X + PLAY_WIDTH + 30 # Consistent with score's sx
    sy = TOP_LEFT_Y + PLAY_HEIGHT/2 - 150 # Adjusted Y to move it up
    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, piece.color, (sx + j*BLOCK_SIZE, sy + i*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    # Position label above the shape preview
    surface.blit(label, (sx + (5*BLOCK_SIZE - label.get_width())/2, sy - label.get_height() - 5))


def draw_window(surface, grid, score=0, last_score = 0, generation=0, population_size=0, genome_id=0):
    surface.fill((0,0,0)) # Black background

    # font = pygame.font.SysFont('comicsans', 60)
    # label = font.render('NEAT TETRIS', 1, (255,255,255))
    # # Lower the title to be above the grid
    # surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH / 2 - (label.get_width() / 2), TOP_LEFT_Y - label.get_height() - 10))

    # Current Score
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Score: ' + str(score), 1, (255,255,255))
    # Adjusted sx for better positioning
    sx = TOP_LEFT_X + PLAY_WIDTH + 30 # Moved slightly left
    sy = TOP_LEFT_Y + PLAY_HEIGHT/2 + 50 # Moved slightly down
    surface.blit(label, (sx, sy + 150)) # Adjusted y for score

    # Last Score (Max Score)
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Max Score: ' + str(last_score), 1, (255,255,255))
    # Adjusted x and y position to be to the left of the grid
    max_score_x = TOP_LEFT_X - label.get_width() - 30 # Moved to the left of the grid
    max_score_y = TOP_LEFT_Y + PLAY_HEIGHT/2 + 50 # Aligned with score vertically
    surface.blit(label, (max_score_x, max_score_y))

    # Generation, Population, Genome ID display (Can be adjusted as needed)
    font = pygame.font.SysFont('comicsans', 20)
    gen_text = f"Gen: {generation}"
    pop_text = f"Pop: {population_size}"
    genome_text = f"Genome: {genome_id}"
    
    gen_label = font.render(gen_text, 1, (255,255,255))
    pop_label = font.render(pop_text, 1, (255,255,255))
    genome_label = font.render(genome_text, 1, (255,255,255))

    # Position these texts as desired, for example, under the max score
    text_start_y = max_score_y + label.get_height() + 10
    surface.blit(gen_label, (max_score_x, text_start_y))
    surface.blit(pop_label, (max_score_x, text_start_y + gen_label.get_height() + 5))
    surface.blit(genome_label, (max_score_x, text_start_y + gen_label.get_height() + pop_label.get_height() + 10))


    for r in range(len(grid)):
        for c in range(len(grid[r])):
            pygame.draw.rect(surface, grid[r][c],
                             (TOP_LEFT_X + c*BLOCK_SIZE, TOP_LEFT_Y + r*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    # Draw border and grid lines
    draw_grid_lines(surface, grid)
    pygame.draw.rect(surface, (255,0,0), (TOP_LEFT_X, TOP_LEFT_Y, PLAY_WIDTH, PLAY_HEIGHT), 5)


# --- Game State Variables ---
# These will be managed per genome in the NEAT loop

# --- NEAT Specific Functions ---

def get_board_features(grid, current_piece):
    """
    Calculates features of the board for the AI.
    This is a CRITICAL function. Good features make training easier.
    """
    # 1. Heights of each column
    heights = [0] * 10
    for c in range(10):
        for r in range(20):
            if grid[r][c] != (0,0,0):
                heights[c] = 20 - r
                break
    
    # 2. Aggregate Height (sum of all column heights)
    agg_height = sum(heights)

    # 3. Holes (empty spaces below a filled space in the same column)
    holes = 0
    for c in range(10):
        col_has_block = False
        for r in range(20):
            if grid[r][c] != (0,0,0):
                col_has_block = True
            elif col_has_block and grid[r][c] == (0,0,0):
                holes += 1
    
    # 4. Bumpiness (sum of absolute differences between adjacent column heights)
    bumpiness = 0
    for i in range(9):
        bumpiness += abs(heights[i] - heights[i+1])

    # 5. Lines that would be cleared if current piece is placed (hard to calculate without placing)
    #    For simplicity, we might not use this directly as an input, but use it in fitness.
    #    Instead, let's consider the current piece's y position and type.

    # 6. Current piece type (one-hot encode or index)
    piece_type_idx = shapes.index(current_piece.shape)

    # 7. Current piece rotation
    piece_rotation = current_piece.rotation % len(current_piece.shape)
    
    # Combine features into a flat list
    # Max height of any column
    max_height = max(heights) if heights else 0

    features = heights + [agg_height, holes, bumpiness, max_height, piece_type_idx, piece_rotation]
    # Total: 10 (heights) + 1 (agg_h) + 1 (holes) + 1 (bump) + 1 (max_h) + 1 (piece_idx) + 1 (piece_rot) = 16 features
    return features


def simulate_move(piece, grid, move_x, rotation_target):
    """
    Simulates placing a piece with a given horizontal move and rotation.
    Returns the resulting grid state (or features of it) and if the move is valid.
    This is a simplified simulation: it tries to rotate, then move horizontally, then drops.
    """
    temp_piece = Piece(piece.x, piece.y, piece.shape)
    temp_piece.rotation = piece.rotation # Start with current rotation

    # 1. Simulate Rotation
    for _ in range(rotation_target - (temp_piece.rotation % len(temp_piece.shape))): # Number of rotations needed
        temp_piece.rotation += 1
        if not valid_space(temp_piece, grid):
            temp_piece.rotation -=1 # Revert if invalid rotation
            # If rotation is immediately invalid, this sequence might be bad
            # For simplicity, we'll proceed, but more advanced AI would penalize this.
            break 

    # 2. Simulate Horizontal Move
    temp_piece.x += move_x
    if not valid_space(temp_piece, grid):
        # If horizontal move is invalid, this specific (rotation, x_offset) is bad
        return None, 0, 0, 0, 0 # Invalid move indicator

    # 3. Simulate Drop (Hard Drop)
    while valid_space(temp_piece, grid):
        temp_piece.y += 1
    temp_piece.y -= 1 # Go back to last valid position

    # Create a temporary grid with the piece locked
    temp_locked_positions = {}
    for r_idx in range(len(grid)):
        for c_idx in range(len(grid[0])):
            if grid[r_idx][c_idx] != (0,0,0):
                temp_locked_positions[(c_idx, r_idx)] = grid[r_idx][c_idx]

    formatted_shape = convert_shape_format(temp_piece)
    for pos in formatted_shape:
        if pos[1] < 0: # Piece locked above screen -> game over for this sim
            return None, 0, 0, 0, 0 # This would be a game over state
        temp_locked_positions[(pos[0], pos[1])] = temp_piece.color
    
    temp_grid_after_move = create_grid(temp_locked_positions)

    # Calculate features of this temporary grid
    # We need features specifically about the outcome of *this* move
    
    # a. Lines cleared by this move
    lines_cleared_by_move = 0
    full_rows = 0
    for r in range(19, -1, -1): # Check from bottom up
        is_full = True
        for c in range(10):
            if temp_grid_after_move[r][c] == (0,0,0):
                is_full = False
                break
        if is_full:
            full_rows +=1
    lines_cleared_by_move = full_rows

    # b. Heights after move
    heights_after = [0] * 10
    for c in range(10):
        for r in range(20):
            if temp_grid_after_move[r][c] != (0,0,0):
                heights_after[c] = 20 - r
                break
    
    # c. Aggregate height after move
    agg_height_after = sum(heights_after)

    # d. Holes after move
    holes_after = 0
    for c in range(10):
        col_has_block_after = False
        for r_idx in range(20):
            if temp_grid_after_move[r_idx][c] != (0,0,0):
                col_has_block_after = True
            elif col_has_block_after and temp_grid_after_move[r_idx][c] == (0,0,0):
                holes_after += 1
                
    # e. Bumpiness after move
    bumpiness_after = 0
    for i in range(9):
        bumpiness_after += abs(heights_after[i] - heights_after[i+1])

    return temp_grid_after_move, lines_cleared_by_move, agg_height_after, holes_after, bumpiness_after


# --- Main Game Loop for NEAT ---
MAX_SCORE_SO_FAR = 0
GENERATION_COUNT = 0

# --- Function to Draw Neural Network ---
def draw_neural_network(surface, genome, config, x_start, y_start, width, height):
    """
    Draws a representation of the neural network.
    """
    # ADDED FOR DEBUGGING
    # print("--- Drawing NN Debug Info ---")
    if not genome or not config:
        # print("Genome or Config is None. Exiting draw_neural_network.")
        # Fallback to draw an error message if no genome/config
        font = pygame.font.SysFont('comicsans', 16)
        error_label = font.render('No genome data', 1, (150,150,150))
        pygame.draw.rect(surface, (30, 30, 30), (x_start, y_start, width, height)) # Background
        surface.blit(error_label, (x_start + (width - error_label.get_width())/2, y_start + (height - error_label.get_height())/2))
        return
    # else:
    #     dbg_input_keys_print = list(config.genome_config.input_keys) # Renamed for clarity
    #     dbg_output_keys_print = list(config.genome_config.output_keys) # Renamed for clarity
    #     dbg_genome_node_keys_print = list(genome.nodes.keys()) if genome and hasattr(genome, 'nodes') else 'N/A'
    #     dbg_connections_print = [(cg.key, cg.weight, cg.enabled) for cg_key, cg in genome.connections.items()] if genome and hasattr(genome, 'connections') else 'N/A'
        
    #     print(f"  Config Input Keys: {dbg_input_keys_print}")
    #     print(f"  Config Output Keys: {dbg_output_keys_print}")
    #     print(f"  Genome Node Keys: {dbg_genome_node_keys_print}")
    #     print(f"  Number of enabled connections in genome: {len([c for c in dbg_connections_print if c[2]]) if isinstance(dbg_connections_print, list) else 'N/A'}")

    node_radius = 7
    node_spacing_y_within_layer = 12
    line_thickness_multiplier = 2.5
    padding = 15

    pygame.draw.rect(surface, (30, 30, 30), (x_start, y_start, width, height))
    pygame.draw.rect(surface, (80, 80, 80), (x_start, y_start, width, height), 2)

    font = pygame.font.SysFont('comicsans', 16)
    title_label = font.render('Best Genome NN', 1, (220,220,220))
    title_x = x_start + (width - title_label.get_width()) / 2
    title_y = y_start + padding / 3
    surface.blit(title_label, (title_x, title_y))

    content_y_start = title_y + title_label.get_height() + padding / 2
    content_x_start = x_start + padding
    content_width = width - 2 * padding
    content_height = height - (content_y_start - y_start) - padding

    # Use actual input/output keys from config for drawing
    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    
    # Get actual node genes from the genome, if they exist (for hidden nodes mainly)
    # Input nodes might not be in genome.nodes if they are default.
    genome_nodes_present = genome.nodes if hasattr(genome, 'nodes') else {}

    node_positions = {}

    # Simplified Layering:
    # Layer 0: Input nodes
    # Layer 1: Hidden nodes (if any, from genome.nodes that are not I/O)
    # Layer 2: Output nodes
    
    # print(f"  Using simplified layering. Input keys: {input_keys}, Output keys: {output_keys}")

    potential_hidden_nodes = set(genome_nodes_present.keys()) - set(input_keys) - set(output_keys)
    
    layers_for_drawing = [set(input_keys)]
    if potential_hidden_nodes:
        layers_for_drawing.append(potential_hidden_nodes)
    layers_for_drawing.append(set(output_keys))
    
    layers_for_drawing = [l for l in layers_for_drawing if l] # Filter out empty layers (e.g. no hidden nodes)

    # print(f"  Simplified Layers (after filtering empty): {layers_for_drawing}")


    if not layers_for_drawing:
        no_layers_label = font.render('No layers for drawing', 1, (150,150,150))
        surface.blit(no_layers_label, (x_start + (width - no_layers_label.get_width())/2, y_start + (height - no_layers_label.get_height())/2))
        return

    num_layers = len(layers_for_drawing)
    if num_layers == 0: return

    layer_x_positions = []
    if num_layers == 1:
        layer_x_positions.append(content_x_start + content_width / 2)
    else:
        spacing = (content_width - 2 * node_radius) / (num_layers - 1) if num_layers > 1 else 0
        for i in range(num_layers):
            layer_x_positions.append(content_x_start + node_radius + i * spacing)

    for i, layer_nodes_set in enumerate(layers_for_drawing):
        layer_nodes = sorted(list(layer_nodes_set)) # Sort for consistent drawing order
        if not layer_nodes: continue

        num_nodes_in_layer = len(layer_nodes)
        current_layer_x = layer_x_positions[i]
        
        # Dynamic vertical spacing based on available height and number of nodes
        total_node_height = num_nodes_in_layer * (2 * node_radius)
        if num_nodes_in_layer > 1:
            effective_spacing_y = (content_height - total_node_height) / (num_nodes_in_layer -1)
            effective_spacing_y = max(effective_spacing_y, node_spacing_y_within_layer / 2) # Min spacing
            effective_spacing_y = min(effective_spacing_y, node_spacing_y_within_layer * 1.5) # Max cap to prevent too much spread
        else:
            effective_spacing_y = 0

        start_y_for_layer = (content_height - (total_node_height + max(0, num_nodes_in_layer - 1) * effective_spacing_y)) / 2
        start_y_for_layer = max(start_y_for_layer, 0) # Ensure it's not negative

        for node_idx, node_id in enumerate(layer_nodes):
            y_pos = content_y_start + start_y_for_layer + node_idx * (2 * node_radius + effective_spacing_y) + node_radius
            node_positions[node_id] = (current_layer_x, y_pos)

    # Draw Connections
    if hasattr(genome, 'connections'):
        for cg_key, cg_gene in genome.connections.items(): # cg_gene is ConnectionGene object
            if cg_gene.enabled:
                input_node_id, output_node_id = cg_gene.key
                pos_in = node_positions.get(input_node_id)
                pos_out = node_positions.get(output_node_id)

                if pos_in and pos_out:
                    weight = cg_gene.weight
                    if weight > 0:
                        intensity = int(min(255, max(30, 50 + weight * 150)))
                        color = (0, intensity, 0)
                    else:
                        intensity = int(min(255, max(30, 50 + abs(weight) * 150)))
                        color = (intensity, 0, 0)
                    
                    thickness = int(max(1, min(abs(weight * line_thickness_multiplier), 3)))
                    try:
                        pygame.draw.line(surface, color, pos_in, pos_out, thickness)
                    except TypeError:
                        pass # Should be rare if pos_in/pos_out are good

    # Draw Nodes
    # print(f"  Node positions calculated for keys: {list(node_positions.keys())}")
    # print("--- End NN Debug Info ---") # If re-enabling debug

    for node_id, pos in node_positions.items():
        node_color = (180, 180, 180) 
        node_border_color = (50,50,50)
        if node_id in input_keys: # Use the definitive input_keys from config
            node_color = (80, 80, 220) 
            node_border_color = (120,120,255)
        elif node_id in output_keys: # Use the definitive output_keys from config
            node_color = (80, 220, 80)
            node_border_color = (120,255,120)
        # Hidden nodes would be default (180,180,180)

        try:
            pygame.draw.circle(surface, node_color, (int(pos[0]), int(pos[1])), node_radius)
            pygame.draw.circle(surface, node_border_color, (int(pos[0]), int(pos[1])), node_radius, 1)
        except TypeError: 
            pass



def eval_genomes(genomes, config, draw_while_training=True):
    global MAX_SCORE_SO_FAR, GENERATION_COUNT, BEST_GENOME_EVER_FOR_VIZ, HIGHEST_FITNESS_EVER_FOR_VIZ
    GENERATION_COUNT += 1
    win = None # Initialize win to None
    if draw_while_training:
        win = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
        pygame.display.set_caption(f"NEAT Tetris - Gen: {GENERATION_COUNT}")


    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        # --- Game State Variables (reinitialize for each genome) ---
        locked_positions = {}  # (x,y): (r,g,b)
        grid = create_grid(locked_positions)
        change_piece = False
        run = True
        current_piece = get_shape()
        next_piece = get_shape()
        clock = pygame.time.Clock()
        fall_time = 0
        fall_speed = 0.27 # How fast piece falls (seconds per step) - AI can play fast
        level_time = 0
        score = 0
        pieces_placed = 0

        # AI specific counters for fitness
        total_lines_cleared_genome = 0
        
        game_frames = 0 # For survival fitness

        while run:
            game_frames += 1
            grid = create_grid(locked_positions) # Update grid based on locked pieces
            fall_time += clock.get_rawtime()
            level_time += clock.get_rawtime()
            clock.tick() # Control game speed

            # --- AI Decision Making ---
            # The AI will try all possible moves for the current piece
            # and pick the one its network scores highest.
            
            best_move_score = -float('inf')
            best_move_details = None # (target_rotation, target_x_final_column)

            # Inputs for the NN: features *resulting from a potential move*
            # Outputs from the NN: a single score for that potential move.

            possible_final_positions = [] # Store (final_x, final_rotation, resulting_features_tuple)

            for r_idx in range(len(current_piece.shape)): # Try all rotations
                # Create a temporary piece for this rotation
                sim_piece_rot = Piece(current_piece.x, current_piece.y, current_piece.shape)
                sim_piece_rot.rotation = r_idx
                
                # Try all possible columns for this rotation
                for c_idx in range(-5, 15): # Check a wide range of x, valid_space will clip
                    sim_piece_drop = Piece(c_idx, 0, sim_piece_rot.shape) # Start at top
                    sim_piece_drop.rotation = sim_piece_rot.rotation
                    
                    # Check if this horizontal position is initially valid for the rotated piece
                    # (before dropping)
                    if not valid_space(sim_piece_drop, grid):
                        continue

                    # Simulate dropping this piece
                    temp_y = sim_piece_drop.y
                    while valid_space(sim_piece_drop, grid):
                        sim_piece_drop.y += 1
                    sim_piece_drop.y -= 1
                    
                    if sim_piece_drop.y < temp_y: # Could not even drop one step from initial
                        continue

                    # Now, `sim_piece_drop` is at its lowest possible y for this (c_idx, r_idx)
                    # Create a temporary grid with this piece placed
                    temp_locked = locked_positions.copy()
                    formatted = convert_shape_format(sim_piece_drop)
                    
                    valid_placement = True
                    for pos in formatted:
                        if pos[1] < 0: # Part of piece is above screen when "locked"
                            valid_placement = False
                            break
                        temp_locked[(pos[0], pos[1])] = sim_piece_drop.color
                    
                    if not valid_placement:
                        continue

                    temp_grid_after_move = create_grid(temp_locked)
                    
                    # Calculate features of this *resulting* board state
                    heights_after = [0] * 10
                    for c_col in range(10):
                        for r_row in range(20):
                            if temp_grid_after_move[r_row][c_col] != (0,0,0):
                                heights_after[c_col] = 20 - r_row
                                break
                    
                    agg_height_after = sum(heights_after)
                    
                    holes_after = 0
                    for c_col in range(10):
                        col_has_block_after = False
                        for r_row in range(20):
                            if temp_grid_after_move[r_row][c_col] != (0,0,0):
                                col_has_block_after = True
                            elif col_has_block_after and temp_grid_after_move[r_row][c_col] == (0,0,0):
                                holes_after += 1
                                
                    bumpiness_after = 0
                    for i in range(9):
                        bumpiness_after += abs(heights_after[i] - heights_after[i+1])

                    # Lines that *would be* cleared by THIS move
                    lines_cleared_by_this_move = 0
                    _temp_locked_for_clear = temp_locked.copy() # Use a copy to simulate clearing
                    
                    # Simulate clear_rows to count
                    num_cleared_sim = 0
                    for r_clear_idx in range(19, -1, -1):
                        is_full = True
                        for c_clear_idx in range(10):
                            if temp_grid_after_move[r_clear_idx][c_clear_idx] == (0,0,0):
                                is_full = False
                                break
                        if is_full:
                            num_cleared_sim +=1
                    lines_cleared_by_this_move = num_cleared_sim
                    
                    # Input features for the network for *this specific move*
                    # These should describe the state *after* the move.
                    # Order: agg_height, completed_lines, holes, bumpiness
                    move_inputs = (
                        agg_height_after,
                        lines_cleared_by_this_move, # crucial
                        holes_after,
                        bumpiness_after
                    )
                    
                    # Get NN's evaluation of this move
                    output = net.activate(move_inputs)
                    move_score = output[0] # Assuming single output node

                    if move_score > best_move_score:
                        best_move_score = move_score
                        best_move_details = (sim_piece_drop.x, sim_piece_drop.rotation, sim_piece_drop.y) # Store final x, rotation, and final y
            
            # Execute the best move found
            if best_move_details:
                target_x, target_rotation, target_y = best_move_details
                current_piece.x = target_x
                current_piece.rotation = target_rotation
                current_piece.y = target_y # Move directly to final y (hard drop)
                
                # Lock piece immediately after "best move" is decided
                change_piece = True # Since AI plays optimally, it places the piece
            else:
                # No valid move found (should be rare, might mean game over is imminent or bug)
                # genome.fitness -= 10 # Penalize heavily
                run = False # End this genome's run
                # print(f"Genome {genome_id} found no valid move. Ending run.")
                continue


            # --- Normal Game Logic for Piece Locking (after AI chose best drop) ---
            # This section is reached if a piece is to be locked
            if change_piece:
                shape_pos = convert_shape_format(current_piece)
                for pos in shape_pos:
                    p = (pos[0], pos[1])
                    locked_positions[p] = current_piece.color
                
                current_piece = next_piece
                next_piece = get_shape()
                change_piece = False
                pieces_placed += 1

                cleared_rows_count = clear_rows(grid, locked_positions)
                total_lines_cleared_genome += cleared_rows_count
                
                # Update score (can be part of fitness too)
                if cleared_rows_count == 1: score += 40
                elif cleared_rows_count == 2: score += 100
                elif cleared_rows_count == 3: score += 300
                elif cleared_rows_count == 4: score += 1200 # Tetris!

                # Fitness calculation for this genome (incrementally or at the end)
                genome.fitness += cleared_rows_count * 10 # Strong reward for clearing lines
                genome.fitness += 0.1 # Small reward for surviving longer / placing pieces

                if score > MAX_SCORE_SO_FAR:
                    MAX_SCORE_SO_FAR = score
            
            # Check if lost
            if check_lost(locked_positions):
                # Penalize based on how bad the final state is (e.g. height)
                heights = [0] * 10
                for c in range(10):
                    for r in range(20):
                        if grid[r][c] != (0,0,0):
                            heights[c] = 20 - r
                            break
                genome.fitness -= sum(heights) * 0.1 # Penalize high stacks at game over
                run = False
                # print(f"Genome {genome_id} lost. Score: {score}, Lines: {total_lines_cleared_genome}, Fitness: {genome.fitness:.2f}")

            # Update BEST_GENOME_EVER_FOR_VIZ and HIGHEST_FITNESS_EVER_FOR_VIZ
            if genome.fitness is not None and genome.fitness > HIGHEST_FITNESS_EVER_FOR_VIZ:
                HIGHEST_FITNESS_EVER_FOR_VIZ = genome.fitness
                BEST_GENOME_EVER_FOR_VIZ = genome


            # --- Drawing ---
            if draw_while_training:
                # Only draw if you want to watch it train (slows down significantly)
                # For faster training, comment out drawing or draw only for best genome.
                # if GENERATION_COUNT % 5 == 0 and genome_id == genomes[0][0]: # Draw first genome every 5 gens
                # if True: # Draw all # This line would be controlled by draw_while_training
                draw_window(win, grid, score, MAX_SCORE_SO_FAR, GENERATION_COUNT, len(genomes), genome_id)
                draw_next_shape(next_piece, win)

                # Draw the best neural network found so far
                next_shape_display_sx = TOP_LEFT_X + PLAY_WIDTH + 30
                next_shape_display_width = 5 * BLOCK_SIZE
                padding_after_next_shape = 20
                padding_right_edge = 20

                nn_area_x = next_shape_display_sx + next_shape_display_width + padding_after_next_shape
                nn_area_y = TOP_LEFT_Y + 50
                nn_area_width = S_WIDTH - nn_area_x - padding_right_edge
                nn_area_height = PLAY_HEIGHT - 200
                
                if BEST_GENOME_EVER_FOR_VIZ:
                    if nn_area_width > 0:
                         draw_neural_network(win, BEST_GENOME_EVER_FOR_VIZ, config, nn_area_x, nn_area_y, nn_area_width, nn_area_height)

                temp_piece_draw = Piece(current_piece.x, current_piece.y, current_piece.shape)
                temp_piece_draw.rotation = current_piece.rotation
                temp_piece_draw.color = current_piece.color
                formatted_shape_draw = convert_shape_format(temp_piece_draw)
                for pos in formatted_shape_draw:
                    pygame.draw.rect(win, temp_piece_draw.color,
                                     (TOP_LEFT_X + pos[0]*BLOCK_SIZE, TOP_LEFT_Y + pos[1]*BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 0)
                pygame.display.update()

            # --- Event Handling (for quitting, not for AI control) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit() # Exit entire program
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: # Add a quit key
                        pygame.quit()
                        quit()

            # Limit max game duration for a single genome to prevent infinite loops
            if game_frames > 2000 + (total_lines_cleared_genome * 50): # Allow more frames if clearing lines
                # print(f"Genome {genome_id} timed out. Fitness: {genome.fitness:.2f}")
                run = False


def run_neat(config_path, draw_while_training=True):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)

    # Add reporters for output
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5)) # Save checkpoints every 5 generations

    # Use a lambda to pass the draw_while_training argument to eval_genomes
    winner = p.run(lambda genomes, config_obj: eval_genomes(genomes, config_obj, draw_while_training), 50)


    # Save the winner
    # Also save the config with the winner for later loading if needed
    with open('best_tetris_ai.pkl', 'wb') as output:
        pickle.dump((winner, config), output, 1) # Save (winner_genome, config_object)

    print('\nBest genome:\n{!s}'.format(winner))
    
    # Visualize the best genome playing (optional)
    # You'd need a separate function similar to eval_genomes but just for one genome
    # and without fitness updates, just playing.


def play_with_ai(genome_path, config_path=None):
    # Config path is now optional if config is saved with genome
    loaded_config = None
    with open(genome_path, "rb") as f:
        data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 2: # New format: (genome, config)
            genome, loaded_config = data
        else: # Old format: just genome
            genome = data
            print("Warning: Loaded genome might be from an old save without config. Loading from default config_path.")

    if loaded_config: # Use the config saved with the genome
        config = loaded_config
    elif config_path: # Fallback to provided config_path
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_path)
    else:
        print("Error: Cannot load AI. Config not found with genome and no config_path provided.")
        return

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    win = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
    
    locked_positions = {}
    grid = create_grid(locked_positions)
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    score = 0
    run = True
    
    print("Playing with the best AI. Press Q to quit.")

    while run:
        grid = create_grid(locked_positions)
        clock.tick(10) # Play at a visible speed

        # AI Decision Making (same as in eval_genomes)
        best_move_score = -float('inf')
        best_move_details = None

        for r_idx in range(len(current_piece.shape)):
            sim_piece_rot = Piece(current_piece.x, current_piece.y, current_piece.shape)
            sim_piece_rot.rotation = r_idx
            for c_idx in range(-5, 15):
                sim_piece_drop = Piece(c_idx, 0, sim_piece_rot.shape)
                sim_piece_drop.rotation = sim_piece_rot.rotation
                if not valid_space(sim_piece_drop, grid): continue

                temp_y = sim_piece_drop.y
                while valid_space(sim_piece_drop, grid): sim_piece_drop.y += 1
                sim_piece_drop.y -= 1
                if sim_piece_drop.y < temp_y: continue

                temp_locked = locked_positions.copy()
                formatted = convert_shape_format(sim_piece_drop)
                valid_placement = True
                for pos in formatted:
                    if pos[1] < 0: valid_placement = False; break
                    temp_locked[(pos[0], pos[1])] = sim_piece_drop.color
                if not valid_placement: continue
                
                temp_grid_after_move = create_grid(temp_locked)
                heights_after = [0] * 10
                for c_col in range(10):
                    for r_row in range(20):
                        if temp_grid_after_move[r_row][c_col] != (0,0,0):
                            heights_after[c_col] = 20 - r_row; break
                agg_height_after = sum(heights_after)
                holes_after = 0
                for c_col in range(10):
                    col_has_block_after = False
                    for r_row in range(20):
                        if temp_grid_after_move[r_row][c_col] != (0,0,0): col_has_block_after = True
                        elif col_has_block_after and temp_grid_after_move[r_row][c_col] == (0,0,0): holes_after += 1
                bumpiness_after = 0
                for i in range(9): bumpiness_after += abs(heights_after[i] - heights_after[i+1])
                
                lines_cleared_by_this_move = 0
                num_cleared_sim = 0
                for r_clear_idx in range(19, -1, -1):
                    is_full = True
                    for c_clear_idx in range(10):
                        if temp_grid_after_move[r_clear_idx][c_clear_idx] == (0,0,0):
                            is_full = False; break
                    if is_full: num_cleared_sim +=1
                lines_cleared_by_this_move = num_cleared_sim
                
                move_inputs = (agg_height_after, lines_cleared_by_this_move, holes_after, bumpiness_after)
                output = net.activate(move_inputs)
                move_score = output[0]
                if move_score > best_move_score:
                    best_move_score = move_score
                    best_move_details = (sim_piece_drop.x, sim_piece_drop.rotation, sim_piece_drop.y)
        
        if best_move_details:
            target_x, target_rotation, target_y = best_move_details
            current_piece.x = target_x
            current_piece.rotation = target_rotation
            current_piece.y = target_y # Hard drop

            shape_pos = convert_shape_format(current_piece)
            for pos in shape_pos:
                locked_positions[(pos[0], pos[1])] = current_piece.color
            
            current_piece = next_piece
            next_piece = get_shape()
            cleared_rows_count = clear_rows(grid, locked_positions) # grid updated inside clear_rows by modifying locked_positions
            if cleared_rows_count == 1: score += 40
            elif cleared_rows_count == 2: score += 100
            elif cleared_rows_count == 3: score += 300
            elif cleared_rows_count == 4: score += 1200
        else:
            print("AI found no valid move. Game Over.")
            run = False

        if check_lost(locked_positions):
            print(f"Game Over! Final Score: {score}")
            run = False

        draw_window(win, grid, score, MAX_SCORE_SO_FAR)
        draw_next_shape(next_piece, win)
        # Draw current piece
        temp_piece_draw = Piece(current_piece.x, current_piece.y, current_piece.shape)
        temp_piece_draw.rotation = current_piece.rotation
        temp_piece_draw.color = current_piece.color
        formatted_shape_draw = convert_shape_format(temp_piece_draw)
        for pos in formatted_shape_draw:
            pygame.draw.rect(win, temp_piece_draw.color,
                                (TOP_LEFT_X + pos[0]*BLOCK_SIZE, TOP_LEFT_Y + pos[1]*BLOCK_SIZE,
                                BLOCK_SIZE, BLOCK_SIZE), 0)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    run = False
    
    pygame.quit()


def display_main_menu(surface, save_file_path):
    font = pygame.font.SysFont('arial', 28, bold=True)
    small_font = pygame.font.SysFont('arial', 18)
    
    button_width = 480
    button_height = 70
    button_spacing = 25
    total_button_height = (button_height * 3) + (button_spacing * 2)
    start_y = (S_HEIGHT - total_button_height) // 2 + 40
    button_x = (S_WIDTH - button_width) // 2

    button_train_draw_rect = pygame.Rect(button_x, start_y, button_width, button_height)
    button_train_no_draw_rect = pygame.Rect(button_x, start_y + button_height + button_spacing, button_width, button_height)
    button_play_saved_rect = pygame.Rect(button_x, start_y + 2 * (button_height + button_spacing), button_width, button_height)

    model_exists = os.path.exists(save_file_path)

    # Modern color scheme
    colors = {
        "background": (15, 15, 25),
        "button_normal": (45, 55, 85),
        "button_hover": (60, 75, 115),
        "button_disabled": (40, 40, 50),
        "button_border": (80, 100, 140),
        "button_border_hover": (100, 130, 180),
        "button_border_disabled": (60, 60, 70),
        "text_normal": (255, 255, 255),
        "text_disabled": (120, 120, 130),
        "title_color": (220, 230, 255),
        "subtitle_color": (160, 170, 200),
        "shadow": (0, 0, 0, 100)
    }

    title_font = pygame.font.SysFont('arial', 64, bold=True)
    subtitle_font = pygame.font.SysFont('arial', 20)
    title_label = title_font.render('NEAT TETRIS AI', True, colors["title_color"])
    subtitle_label = subtitle_font.render('Choose your training mode', True, colors["subtitle_color"])

    clock = pygame.time.Clock()

    def draw_button_with_shadow(surface, rect, bg_color, border_color, text, text_color, is_hovered=False):
        # Draw shadow
        shadow_rect = pygame.Rect(rect.x + 3, rect.y + 3, rect.width, rect.height)
        shadow_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0, 0, 0, 80), (0, 0, rect.width, rect.height), border_radius=12)
        surface.blit(shadow_surface, shadow_rect.topleft)
        
        # Draw button background
        pygame.draw.rect(surface, bg_color, rect, border_radius=12)
        
        # Draw border with gradient effect
        border_width = 3 if is_hovered else 2
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=12)
        
        # Add subtle inner highlight
        if is_hovered:
            highlight_rect = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height // 3)
            highlight_surface = pygame.Surface((highlight_rect.width, highlight_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surface, (255, 255, 255, 20), (0, 0, highlight_rect.width, highlight_rect.height), border_radius=8)
            surface.blit(highlight_surface, highlight_rect.topleft)
        
        # Draw text
        text_surface = font.render(text, True, text_color)
        text_x = rect.centerx - text_surface.get_width() // 2
        text_y = rect.centery - text_surface.get_height() // 2
        surface.blit(text_surface, (text_x, text_y))

    while True:
        # Create gradient background
        for y in range(S_HEIGHT):
            color_ratio = y / S_HEIGHT
            r = int(colors["background"][0] * (1 - color_ratio * 0.3))
            g = int(colors["background"][1] * (1 - color_ratio * 0.3))
            b = int(colors["background"][2] + color_ratio * 15)
            pygame.draw.line(surface, (r, g, b), (0, y), (S_WIDTH, y))

        # Draw title with shadow
        title_shadow_x = S_WIDTH//2 - title_label.get_width()//2 + 2
        title_shadow_y = start_y - title_label.get_height() - 80 + 2
        title_shadow = title_font.render('NEAT TETRIS AI', True, (0, 0, 0))
        surface.blit(title_shadow, (title_shadow_x, title_shadow_y))
        
        title_x = S_WIDTH//2 - title_label.get_width()//2
        title_y = start_y - title_label.get_height() - 80
        surface.blit(title_label, (title_x, title_y))
        
        # Draw subtitle
        subtitle_x = S_WIDTH//2 - subtitle_label.get_width()//2
        subtitle_y = title_y + title_label.get_height() + 10
        surface.blit(subtitle_label, (subtitle_x, subtitle_y))

        mouse_pos = pygame.mouse.get_pos()

        # Button 1: Train and Draw
        is_hovered_1 = button_train_draw_rect.collidepoint(mouse_pos)
        button_color_1 = colors["button_hover"] if is_hovered_1 else colors["button_normal"]
        border_color_1 = colors["button_border_hover"] if is_hovered_1 else colors["button_border"]
        draw_button_with_shadow(surface, button_train_draw_rect, button_color_1, border_color_1, 
                               'Train AI (with Visualization)', colors["text_normal"], is_hovered_1)

        # Button 2: Train without Drawing
        is_hovered_2 = button_train_no_draw_rect.collidepoint(mouse_pos)
        button_color_2 = colors["button_hover"] if is_hovered_2 else colors["button_normal"]
        border_color_2 = colors["button_border_hover"] if is_hovered_2 else colors["button_border"]
        draw_button_with_shadow(surface, button_train_no_draw_rect, button_color_2, border_color_2,
                               'Train AI (No Visualization - Faster)', colors["text_normal"], is_hovered_2)

        # Button 3: Play Saved Model
        can_click_play = model_exists
        is_hovered_3 = button_play_saved_rect.collidepoint(mouse_pos) and can_click_play
        
        if not model_exists:
            button_color_3 = colors["button_disabled"]
            border_color_3 = colors["button_border_disabled"]
            text_color_3 = colors["text_disabled"]
        else:
            button_color_3 = colors["button_hover"] if is_hovered_3 else colors["button_normal"]
            border_color_3 = colors["button_border_hover"] if is_hovered_3 else colors["button_border"]
            text_color_3 = colors["text_normal"]
        
        draw_button_with_shadow(surface, button_play_saved_rect, button_color_3, border_color_3,
                               'Play with Saved AI', text_color_3, is_hovered_3)
        
        # Draw status message for disabled button
        if not model_exists:
            status_msg = small_font.render('(No saved AI model found)', True, colors["text_disabled"])
            status_x = button_play_saved_rect.centerx - status_msg.get_width() // 2
            status_y = button_play_saved_rect.bottom + 12
            surface.blit(status_msg, (status_x, status_y))

        # Draw instructions at bottom
        instruction_text = "Press Q to quit • Click a button to continue"
        instruction_surface = small_font.render(instruction_text, True, colors["subtitle_color"])
        instruction_x = S_WIDTH//2 - instruction_surface.get_width()//2
        instruction_y = S_HEIGHT - 40
        surface.blit(instruction_surface, (instruction_x, instruction_y))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if button_train_draw_rect.collidepoint(mouse_pos):
                        return "train_draw"
                    if button_train_no_draw_rect.collidepoint(mouse_pos):
                        return "train_no_draw"
                    if can_click_play and button_play_saved_rect.collidepoint(mouse_pos):
                        return "play_saved"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return "quit"

        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    best_ai_path = os.path.join(local_dir, 'best_tetris_ai.pkl')

    # Initialize Pygame modules
    pygame.init() # General Pygame init
    # pygame.font.init() # Already called globally, but doesn't hurt if called again or can be removed if already at top
    # pygame.mixer.init() # Already called globally

    win = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
    pygame.display.set_caption("NEAT Tetris AI")

    running_menu = True
    action_taken = False # To prevent re-showing menu if an action was taken

    while running_menu:
        choice = display_main_menu(win, best_ai_path)

        if choice == "train_draw":
            print("Starting training with game drawing...")
            run_neat(config_path, draw_while_training=True)
            action_taken = True
            if os.path.exists(best_ai_path):
                print("Training finished. Playing with the best AI.")
                play_with_ai(best_ai_path, config_path) # play_with_ai handles its own pygame.quit()
            else:
                print("Training finished, but no AI was saved (or an error occurred).")
            running_menu = False 
        elif choice == "train_no_draw":
            print("Starting training without game drawing (faster)...")
            run_neat(config_path, draw_while_training=False)
            action_taken = True
            if os.path.exists(best_ai_path):
                print("Training finished. Playing with the best AI.")
                play_with_ai(best_ai_path, config_path)
            else:
                print("Training finished, but no AI was saved (or an error occurred).")
            running_menu = False
        elif choice == "play_saved":
            if os.path.exists(best_ai_path):
                print("Playing with saved AI...")
                play_with_ai(best_ai_path, config_path)
                action_taken = True
            else:
                # This case should ideally not be reached if button is properly disabled,
                # but good to have a fallback message.
                print("Saved AI not found. Please train an AI first.")
            running_menu = False 
        elif choice == "quit":
            running_menu = False

    pygame.quit()
    # Using quit() for a cleaner exit if the script is run directly
    # For example, if launched from a terminal, this ensures the process terminates.
    import sys
    sys.exit()