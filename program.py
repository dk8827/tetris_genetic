import pygame
import random
import neat
# import neat.graphs # Not used by the custom NN drawing function
import os
import pickle # To save and load the best genome

# --- Pygame Setup ---
pygame.font.init()
pygame.mixer.init() # For audio effects when needed

# Screen dimensions
S_WIDTH = 1200
S_HEIGHT = 700
PLAY_WIDTH = 300  # 300 // 10 = 30 width per block
PLAY_HEIGHT = 600 # 600 // 20 = 30 height per block
BLOCK_SIZE = 30

TOP_LEFT_X = (S_WIDTH - PLAY_WIDTH) // 2
TOP_LEFT_Y = S_HEIGHT - PLAY_HEIGHT - 50 # 50px padding at the bottom

# --- Global Fonts ---
FONT_COMICSANS_60 = pygame.font.SysFont('comicsans', 60, bold=True)
FONT_COMICSANS_30 = pygame.font.SysFont('comicsans', 30, bold=True)
FONT_COMICSANS_20 = pygame.font.SysFont('comicsans', 20, bold=True)
FONT_COMICSANS_16 = pygame.font.SysFont('comicsans', 16)
# For main menu
FONT_ARIAL_64_BOLD = pygame.font.SysFont('arial', 64, bold=True)
FONT_ARIAL_28_BOLD = pygame.font.SysFont('arial', 28, bold=True)
FONT_ARIAL_20 = pygame.font.SysFont('arial', 20)
FONT_ARIAL_18 = pygame.font.SysFont('arial', 18)


# --- Shapes ---
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

# Pre-calculate relative coordinates for all shapes and rotations
PRECALCULATED_SHAPES = []
for shape_template_list in shapes:
    rotations_for_shape = []
    for shape_matrix_str_list in shape_template_list:
        coords = []
        for i, line_str in enumerate(shape_matrix_str_list):
            for j, char_in_line in enumerate(line_str):
                if char_in_line == '0':
                    coords.append((j - 2, i - 4)) # Offset to center piece around its x,y
        rotations_for_shape.append(tuple(coords))
    PRECALCULATED_SHAPES.append(tuple(rotations_for_shape))


# Global variables for tracking the best genome for visualization
BEST_GENOME_EVER_FOR_VIZ = None
HIGHEST_FITNESS_EVER_FOR_VIZ = -float('inf')

# --- Piece Class ---
class Piece:
    def __init__(self, column, row, shape_idx):
        self.x = column
        self.y = row
        self.shape_idx = shape_idx
        self.num_rotations = len(shapes[self.shape_idx]) # Original shape definition (e.g., S, Z)
        self.color = shape_colors[self.shape_idx]
        self.rotation = 0 # This is rotation state index

    def get_relative_coords(self):
        current_rotation_idx = self.rotation % self.num_rotations
        return PRECALCULATED_SHAPES[self.shape_idx][current_rotation_idx]

# --- Game Logic Functions ---
def create_grid(locked_positions={}):
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)] # 10 cols, 20 rows
    for r_idx in range(len(grid)):
        for c_idx in range(len(grid[r_idx])):
            if (c_idx,r_idx) in locked_positions:
                color = locked_positions[(c_idx,r_idx)]
                grid[r_idx][c_idx] = color
    return grid

def convert_shape_format(piece): # piece is a Piece object
    relative_coords = piece.get_relative_coords()
    return [(piece.x + dx, piece.y + dy) for dx, dy in relative_coords]

def valid_space(piece, grid):
    formatted_shape_abs_coords = convert_shape_format(piece)

    for x_coord, y_coord in formatted_shape_abs_coords:
        # Check grid boundaries
        if not (0 <= x_coord < 10): # Off screen horizontally
            return False
        if y_coord >= 20: # Off screen below
            return False
        # Allow pieces to be above the screen (y_coord < 0)
        if y_coord >= 0: # If on screen or below top edge
            if grid[y_coord][x_coord] != (0,0,0): # Check for collision with existing blocks
                return False
    return True


def check_lost(positions): # positions is locked_positions
    for pos in positions:
        _, y = pos
        if y < 1: # If any locked piece is in the top visible row (row 0) or above (y < 0)
            return True
    return False

def get_shape():
    shape_idx = random.randrange(len(shapes))
    return Piece(5, 0, shape_idx) # Start piece in middle-top, pass shape_idx

def draw_text_middle(text, size, color, surface): # size param means dynamic font creation here is fine
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, 1, color)
    surface.blit(label, (TOP_LEFT_X + PLAY_WIDTH/2 - (label.get_width()/2),
                         TOP_LEFT_Y + PLAY_HEIGHT/2 - label.get_height()/2 - 150))

def draw_grid_lines(surface, grid_rows): # Parameter renamed for clarity
    sx = TOP_LEFT_X
    sy = TOP_LEFT_Y
    # Horizontal lines (for rows)
    for i in range(grid_rows + 1): # Need one more line than rows for bottom border
        pygame.draw.line(surface, (128,128,128), (sx, sy + i*BLOCK_SIZE), (sx + PLAY_WIDTH, sy + i*BLOCK_SIZE))
    # Vertical lines (for columns)
    for j in range(10 + 1): # Need one more line than columns for right border
        pygame.draw.line(surface, (128,128,128), (sx + j*BLOCK_SIZE, sy), (sx + j*BLOCK_SIZE, sy + PLAY_HEIGHT))


def clear_rows(grid, locked): # grid is passed but not directly used for modification logic, locked is key
    increment = 0
    rows_to_clear = []
    for i in range(len(grid)-1, -1, -1): # Iterate from bottom up
        row_is_full = True
        for j in range(len(grid[0])):
            if grid[i][j] == (0,0,0): 
                row_is_full = False
                break
        if row_is_full:
            increment += 1
            rows_to_clear.append(i)
            for j_col in range(len(grid[0])):
                if (j_col, i) in locked:
                    del locked[(j_col, i)]
    
    if increment > 0:
        new_locked = {}
        # Sort keys by y-value to process correctly when shifting
        sorted_locked_keys = sorted(locked.keys(), key=lambda k: k[1], reverse=True) # y-descending

        for x_lock, y_lock in sorted_locked_keys:
            num_cleared_below = sum(1 for r_cleared in rows_to_clear if y_lock < r_cleared)
            new_y = y_lock + num_cleared_below
            new_locked[(x_lock, new_y)] = locked[(x_lock, y_lock)]
        
        locked.clear()
        locked.update(new_locked)
        
    return increment


def draw_next_shape(piece, surface): # piece is next_piece
    label = FONT_COMICSANS_30.render('Next Shape', 1, (255,255,255))

    sx = TOP_LEFT_X + PLAY_WIDTH + 30
    sy = TOP_LEFT_Y + PLAY_HEIGHT/2 - 150
    
    shape_format_strings = shapes[piece.shape_idx][piece.rotation % piece.num_rotations]

    for i, line in enumerate(shape_format_strings): 
        for j, column_char in enumerate(line): 
            if column_char == '0':
                pygame.draw.rect(surface, piece.color, 
                                 (sx + j*BLOCK_SIZE, sy + i*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    surface.blit(label, (sx + (5*BLOCK_SIZE - label.get_width())/2, sy - label.get_height() - 5))


def draw_window(surface, grid, score=0, last_score = 0, generation=0, population_size=0, genome_id=0):
    surface.fill((0,0,0))

    score_label = FONT_COMICSANS_30.render('Score: ' + str(score), 1, (255,255,255))
    sx_info = TOP_LEFT_X + PLAY_WIDTH + 30
    sy_info_start = TOP_LEFT_Y + PLAY_HEIGHT/2 + 50 
    surface.blit(score_label, (sx_info, sy_info_start + 150)) 

    max_score_label = FONT_COMICSANS_30.render('Max Score: ' + str(last_score), 1, (255,255,255))
    max_score_x = TOP_LEFT_X - max_score_label.get_width() - 30
    max_score_y = sy_info_start 
    surface.blit(max_score_label, (max_score_x, max_score_y))

    gen_text = f"Gen: {generation}"
    pop_text = f"Pop: {population_size}"
    genome_text = f"Genome: {genome_id}"
    
    gen_label = FONT_COMICSANS_20.render(gen_text, 1, (255,255,255))
    pop_label = FONT_COMICSANS_20.render(pop_text, 1, (255,255,255))
    genome_label = FONT_COMICSANS_20.render(genome_text, 1, (255,255,255))

    text_start_y = max_score_y + max_score_label.get_height() + 10
    surface.blit(gen_label, (max_score_x, text_start_y))
    surface.blit(pop_label, (max_score_x, text_start_y + gen_label.get_height() + 5))
    surface.blit(genome_label, (max_score_x, text_start_y + gen_label.get_height() + pop_label.get_height() + 10))

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] != (0,0,0): 
                pygame.draw.rect(surface, grid[r][c],
                                 (TOP_LEFT_X + c*BLOCK_SIZE, TOP_LEFT_Y + r*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)

    draw_grid_lines(surface, len(grid)) 
    pygame.draw.rect(surface, (255,0,0), (TOP_LEFT_X, TOP_LEFT_Y, PLAY_WIDTH, PLAY_HEIGHT), 5) # Border


# --- NEAT Specific Functions ---

# --- AI Decision Function ---
def determine_best_ai_move(current_piece_shape_idx, grid, locked_positions, net):
    """
    Determines the best move for the AI given the current piece, grid state, and neural network.

    Args:
        current_piece_shape_idx (int): The shape index of the piece to be placed.
        grid (list of list of tuple): The current visual game grid.
        locked_positions (dict): Dictionary of (x,y) -> color for already locked pieces.
        net (neat.nn.FeedForwardNetwork): The neural network to evaluate moves.

    Returns:
        tuple or None: (final_x_anchor, final_rotation_idx, final_y_anchor) for the best move,
                       or None if no valid (non-losing) move is found.
    """
    best_move_score = -float('inf')
    best_move_details = None  # (final_x_anchor, final_rotation_idx, final_y_anchor)

    sim_piece_template = Piece(0, 0, current_piece_shape_idx)

    for r_idx in range(sim_piece_template.num_rotations):
        sim_piece_template.rotation = r_idx
        
        current_rel_coords = sim_piece_template.get_relative_coords()
        if not current_rel_coords: continue

        min_dx = min(coord[0] for coord in current_rel_coords)
        max_dx = max(coord[0] for coord in current_rel_coords)
        
        for c_anchor_x in range(-min_dx, 10 - max_dx):
            sim_piece_template.x = c_anchor_x
            sim_piece_template.y = 0 
            
            sim_piece_for_drop = Piece(sim_piece_template.x, sim_piece_template.y, sim_piece_template.shape_idx)
            sim_piece_for_drop.rotation = sim_piece_template.rotation

            while valid_space(sim_piece_for_drop, grid):
                sim_piece_for_drop.y += 1
            sim_piece_for_drop.y -= 1

            formatted_dropped_piece = convert_shape_format(sim_piece_for_drop)
            
            hypothetical_locked_positions = locked_positions.copy()
            for x_coord, y_coord in formatted_dropped_piece:
                hypothetical_locked_positions[(x_coord, y_coord)] = sim_piece_for_drop.color 

            if check_lost(hypothetical_locked_positions):
                continue

            temp_grid_with_piece = [row[:] for row in grid] 
            for x_c, y_c in formatted_dropped_piece:
                if 0 <= x_c < 10 and 0 <= y_c < 20: 
                    temp_grid_with_piece[y_c][x_c] = sim_piece_for_drop.color
            
            heights_after = [0] * 10
            for c in range(10):
                for r_h in range(20):
                    if temp_grid_with_piece[r_h][c] != (0,0,0):
                        heights_after[c] = 20 - r_h
                        break 
            
            agg_height_after = sum(heights_after)
            
            holes_after = 0
            for c in range(10):
                col_has_block = False
                for r_h in range(20):
                    if temp_grid_with_piece[r_h][c] != (0,0,0):
                        col_has_block = True
                    elif col_has_block and temp_grid_with_piece[r_h][c] == (0,0,0):
                        holes_after += 1
                                
            bumpiness_after = 0
            for i in range(9):
                bumpiness_after += abs(heights_after[i] - heights_after[i+1])

            lines_cleared_by_this_move = 0
            for r_idx_clear in range(19, -1, -1):
                if all(cell != (0,0,0) for cell in temp_grid_with_piece[r_idx_clear]):
                    lines_cleared_by_this_move += 1
            
            move_inputs = (
                agg_height_after / 200.0,
                lines_cleared_by_this_move / 4.0,
                holes_after / 200.0,
                bumpiness_after / 180.0 
            )
            
            output = net.activate(move_inputs)
            current_move_score = output[0]

            if current_move_score > best_move_score:
                best_move_score = current_move_score
                best_move_details = (sim_piece_for_drop.x, sim_piece_for_drop.rotation, sim_piece_for_drop.y)
    
    return best_move_details


# --- Main Game Loop for NEAT ---
MAX_SCORE_SO_FAR = 0
GENERATION_COUNT = 0

def draw_neural_network(surface, genome, config, x_start, y_start, width, height):
    if not genome or not config:
        font = FONT_COMICSANS_16 # Use preloaded font
        error_label = font.render('No genome data', 1, (150,150,150))
        pygame.draw.rect(surface, (30, 30, 30), (x_start, y_start, width, height))
        surface.blit(error_label, (x_start + (width - error_label.get_width())/2, y_start + (height - error_label.get_height())/2))
        return

    node_radius = 7
    node_spacing_y_within_layer = 12
    line_thickness_multiplier = 2.5
    padding = 15

    pygame.draw.rect(surface, (30, 30, 30), (x_start, y_start, width, height))
    pygame.draw.rect(surface, (80, 80, 80), (x_start, y_start, width, height), 2)

    title_label = FONT_COMICSANS_16.render('Best Genome NN', 1, (220,220,220))
    title_x = x_start + (width - title_label.get_width()) / 2
    title_y = y_start + padding / 3
    surface.blit(title_label, (title_x, title_y))

    content_y_start = title_y + title_label.get_height() + padding / 2
    content_x_start = x_start + padding
    content_width = width - 2 * padding
    content_height = height - (content_y_start - y_start) - padding

    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)
    genome_nodes_present = genome.nodes if hasattr(genome, 'nodes') else {}
    potential_hidden_nodes = set(genome_nodes_present.keys()) - set(input_keys) - set(output_keys)
    
    layers_for_drawing = [set(input_keys)]
    if potential_hidden_nodes: layers_for_drawing.append(potential_hidden_nodes)
    layers_for_drawing.append(set(output_keys))
    layers_for_drawing = [l for l in layers_for_drawing if l]

    if not layers_for_drawing:
        no_layers_label = FONT_COMICSANS_16.render('No layers for drawing', 1, (150,150,150))
        surface.blit(no_layers_label, (x_start + (width - no_layers_label.get_width())/2, y_start + (height - no_layers_label.get_height())/2))
        return

    num_layers = len(layers_for_drawing)
    if num_layers == 0: return

    node_positions = {}
    layer_x_positions = []
    if num_layers == 1: layer_x_positions.append(content_x_start + content_width / 2)
    else:
        spacing = (content_width - 2 * node_radius) / (num_layers - 1) if num_layers > 1 else 0
        for i in range(num_layers): layer_x_positions.append(content_x_start + node_radius + i * spacing)

    for i, layer_nodes_set in enumerate(layers_for_drawing):
        layer_nodes = sorted(list(layer_nodes_set))
        if not layer_nodes: continue
        num_nodes_in_layer = len(layer_nodes)
        current_layer_x = layer_x_positions[i]
        
        total_node_height = num_nodes_in_layer * (2 * node_radius)
        effective_spacing_y = (content_height - total_node_height) / (num_nodes_in_layer -1) if num_nodes_in_layer > 1 else 0
        effective_spacing_y = max(min(effective_spacing_y, node_spacing_y_within_layer * 1.5), node_spacing_y_within_layer / 2 if num_nodes_in_layer > 1 else 0)
        
        start_y_for_layer = (content_height - (total_node_height + max(0, num_nodes_in_layer - 1) * effective_spacing_y)) / 2
        start_y_for_layer = max(start_y_for_layer, 0)

        for node_idx, node_id in enumerate(layer_nodes):
            y_pos = content_y_start + start_y_for_layer + node_idx * (2 * node_radius + effective_spacing_y) + node_radius
            node_positions[node_id] = (current_layer_x, y_pos)

    if hasattr(genome, 'connections'):
        for cg_key, cg_gene in genome.connections.items():
            if cg_gene.enabled:
                input_node_id, output_node_id = cg_gene.key
                pos_in, pos_out = node_positions.get(input_node_id), node_positions.get(output_node_id)
                if pos_in and pos_out:
                    weight = cg_gene.weight
                    intensity = int(min(255, max(30, 50 + abs(weight) * 150)))
                    color = (0, intensity, 0) if weight > 0 else (intensity, 0, 0)
                    thickness = int(max(1, min(abs(weight * line_thickness_multiplier), 3)))
                    try: pygame.draw.line(surface, color, pos_in, pos_out, thickness)
                    except TypeError: pass # Should ideally log or handle if positions are not valid tuples

    for node_id, pos in node_positions.items():
        node_color, node_border_color = (180,180,180), (50,50,50) # Default for hidden
        if node_id in input_keys: node_color, node_border_color = (80,80,220), (120,120,255)
        elif node_id in output_keys: node_color, node_border_color = (80,220,80), (120,255,120)
        try:
            pygame.draw.circle(surface, node_color, (int(pos[0]), int(pos[1])), node_radius)
            pygame.draw.circle(surface, node_border_color, (int(pos[0]), int(pos[1])), node_radius, 1)
        except TypeError: pass # Should ideally log or handle if pos is not a valid tuple


def eval_genomes(genomes, config, draw_while_training=True):
    global MAX_SCORE_SO_FAR, GENERATION_COUNT, BEST_GENOME_EVER_FOR_VIZ, HIGHEST_FITNESS_EVER_FOR_VIZ
    GENERATION_COUNT += 1
    win = None
    if draw_while_training:
        win = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
        pygame.display.set_caption(f"NEAT Tetris - Gen: {GENERATION_COUNT}")

    for genome_id_tuple, genome in genomes: 
        actual_genome_id = genome_id_tuple 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        locked_positions = {}
        grid = create_grid(locked_positions) 
        current_piece = get_shape()
        next_piece = get_shape()
        clock = pygame.time.Clock()
        score = 0
        total_lines_cleared_genome = 0
        game_frames = 0

        run_genome = True
        while run_genome:
            game_frames += 1
            clock.tick() 

            # --- AI: Determine best move ---
            best_move_details = determine_best_ai_move(
                current_piece.shape_idx, grid, locked_positions, net
            )
            
            # --- Execute the best move ---
            if best_move_details:
                current_piece.x, current_piece.rotation, current_piece.y = best_move_details
                
                shape_pos_to_lock = convert_shape_format(current_piece)
                for p_x, p_y in shape_pos_to_lock:
                    # Allow locking pieces partially or fully above the visible grid
                    # check_lost will handle if this results in a game over
                    locked_positions[(p_x, p_y)] = current_piece.color

                grid = create_grid(locked_positions)

                cleared_rows_count = clear_rows(grid, locked_positions) 
                if cleared_rows_count > 0:
                    grid = create_grid(locked_positions) 

                total_lines_cleared_genome += cleared_rows_count
                
                score_map = {1: 40, 2: 100, 3: 300, 4: 1200}
                score += score_map.get(cleared_rows_count, 0)
                
                genome.fitness += cleared_rows_count * 10  
                if cleared_rows_count == 4: genome.fitness += 5 
                genome.fitness += 0.1 

                current_piece = next_piece
                next_piece = get_shape()

                if score > MAX_SCORE_SO_FAR: MAX_SCORE_SO_FAR = score
            else:
                genome.fitness -= 10 
                run_genome = False
                continue

            if check_lost(locked_positions):
                heights_final = [0] * 10
                for c_col in range(10):
                    for r_row in range(20):
                        if grid[r_row][c_col] != (0,0,0):
                            heights_final[c_col] = 20 - r_row; break
                genome.fitness -= sum(heights_final) * 0.05 
                run_genome = False

            if genome.fitness is not None and genome.fitness > HIGHEST_FITNESS_EVER_FOR_VIZ:
                HIGHEST_FITNESS_EVER_FOR_VIZ = genome.fitness
                BEST_GENOME_EVER_FOR_VIZ = genome

            if draw_while_training:
                draw_window(win, grid, score, MAX_SCORE_SO_FAR, GENERATION_COUNT, len(genomes), actual_genome_id)
                draw_next_shape(next_piece, win)
                
                nn_area_x = TOP_LEFT_X + PLAY_WIDTH + 30 + 5*BLOCK_SIZE + 20 
                nn_area_y = TOP_LEFT_Y + 50
                nn_area_width = S_WIDTH - nn_area_x - 20
                nn_area_height = PLAY_HEIGHT - 200
                if BEST_GENOME_EVER_FOR_VIZ and nn_area_width > 50 and nn_area_height > 50:
                    draw_neural_network(win, BEST_GENOME_EVER_FOR_VIZ, config, nn_area_x, nn_area_y, nn_area_width, nn_area_height)

                pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); import sys; sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q: pygame.quit(); import sys; sys.exit()

            if game_frames > 1500 + (total_lines_cleared_genome * 100): 
                run_genome = False


def run_neat(config_path, draw_while_training=True):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5)) # Optional: uncomment to save checkpoints

    winner = p.run(lambda genomes, config_obj: eval_genomes(genomes, config_obj, draw_while_training), 50) # Train for 50 generations

    with open('best_tetris_ai.pkl', 'wb') as output:
        pickle.dump((winner, config), output, 1)
    print('\nBest genome:\n{!s}'.format(winner))


def play_with_ai(genome_path, config_path=None):
    loaded_config = None
    try:
        with open(genome_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], neat.Config):
                genome, loaded_config = data
            else: genome = data 
    except FileNotFoundError:
        print(f"Error: Genome file not found at {genome_path}")
        return
    except Exception as e:
        print(f"Error loading genome: {e}")
        return

    if not loaded_config and config_path:
        try:
            loaded_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        config_path)
        except Exception as e:
            print(f"Error loading config from path {config_path}: {e}")
            return
    elif not loaded_config:
        print("Error: AI config not found with genome and no fallback config_path provided.")
        return
    
    config = loaded_config 
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    win = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
    pygame.display.set_caption("Tetris - AI Playing")
    
    locked_positions = {}
    grid = create_grid(locked_positions)
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    score = 0
    run_ai_play = True
    
    print("Playing with the best AI. Press Q to quit.")

    while run_ai_play:
        grid = create_grid(locked_positions) 
        clock.tick(10) 

        # --- AI Decision Making ---
        best_move_details = determine_best_ai_move(
            current_piece.shape_idx, grid, locked_positions, net
        )
        
        if best_move_details:
            current_piece.x, current_piece.rotation, current_piece.y = best_move_details
            shape_pos_to_lock = convert_shape_format(current_piece)
            for p_x, p_y in shape_pos_to_lock:
                locked_positions[(p_x, p_y)] = current_piece.color # Allow locking above grid
            
            grid = create_grid(locked_positions) 
            cleared_rows_count = clear_rows(grid, locked_positions)
            if cleared_rows_count > 0:
                grid = create_grid(locked_positions)

            score_map = {1: 40, 2: 100, 3: 300, 4: 1200}
            score += score_map.get(cleared_rows_count, 0)
            
            current_piece = next_piece
            next_piece = get_shape()
        else:
            print("AI found no valid move. Game Over.")
            run_ai_play = False

        if check_lost(locked_positions):
            print(f"Game Over! Final Score: {score}")
            run_ai_play = False

        draw_window(win, grid, score, MAX_SCORE_SO_FAR) 
        draw_next_shape(next_piece, win)
        # Optionally, draw the AI's chosen piece placement briefly before it locks
        # (current_piece before it's swapped to next_piece)
        # For now, the hard drop means it locks instantly and is drawn as part of the grid.
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: run_ai_play = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q: run_ai_play = False
    
    pygame.quit()


def display_main_menu(surface, save_file_path):
    button_width, button_height, button_spacing = 480, 70, 25
    total_button_height = (button_height * 3) + (button_spacing * 2)
    start_y = (S_HEIGHT - total_button_height) // 2 + 40
    button_x = (S_WIDTH - button_width) // 2

    buttons = {
        "train_draw": pygame.Rect(button_x, start_y, button_width, button_height),
        "train_no_draw": pygame.Rect(button_x, start_y + button_height + button_spacing, button_width, button_height),
        "play_saved": pygame.Rect(button_x, start_y + 2 * (button_height + button_spacing), button_width, button_height)
    }
    model_exists = os.path.exists(save_file_path)
    colors = {
        "background": (15,15,25), "button_normal": (45,55,85), "button_hover": (60,75,115),
        "button_disabled": (40,40,50), "button_border": (80,100,140), "button_border_hover": (100,130,180),
        "button_border_disabled": (60,60,70), "text_normal": (255,255,255), "text_disabled": (120,120,130),
        "title_color": (220,230,255), "subtitle_color": (160,170,200)
    }
    title_label = FONT_ARIAL_64_BOLD.render('NEAT TETRIS AI', True, colors["title_color"])
    subtitle_label = FONT_ARIAL_20.render('Choose your action', True, colors["subtitle_color"])
    clock = pygame.time.Clock()

    def draw_button(rect, text, is_hovered, is_enabled):
        bg_color = colors["button_hover"] if is_hovered and is_enabled else (colors["button_normal"] if is_enabled else colors["button_disabled"])
        border_color = colors["button_border_hover"] if is_hovered and is_enabled else (colors["button_border"] if is_enabled else colors["button_border_disabled"])
        text_color = colors["text_normal"] if is_enabled else colors["text_disabled"]

        shadow_rect = pygame.Rect(rect.x + 3, rect.y + 3, rect.width, rect.height)
        shadow_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(shadow_surface, (0,0,0,80), (0,0,rect.width,rect.height), border_radius=12)
        surface.blit(shadow_surface, shadow_rect.topleft)
        pygame.draw.rect(surface, bg_color, rect, border_radius=12)
        pygame.draw.rect(surface, border_color, rect, 3 if is_hovered and is_enabled else 2, border_radius=12)
        
        text_surface = FONT_ARIAL_28_BOLD.render(text, True, text_color)
        surface.blit(text_surface, (rect.centerx - text_surface.get_width()//2, rect.centery - text_surface.get_height()//2))

    while True:
        for y_grad in range(S_HEIGHT): 
            ratio = y_grad / S_HEIGHT
            r,g,b = [int(c*(1-ratio*0.3) if i<2 else c+ratio*15) for i,c in enumerate(colors["background"])]
            pygame.draw.line(surface, (r,g,b), (0,y_grad), (S_WIDTH,y_grad))

        surface.blit(FONT_ARIAL_64_BOLD.render('NEAT TETRIS AI', True, (0,0,0)), (S_WIDTH//2-title_label.get_width()//2+2, start_y-title_label.get_height()-80+2))
        surface.blit(title_label, (S_WIDTH//2-title_label.get_width()//2, start_y-title_label.get_height()-80))
        surface.blit(subtitle_label, (S_WIDTH//2-subtitle_label.get_width()//2, start_y-title_label.get_height()-80+title_label.get_height()+10))

        mouse_pos = pygame.mouse.get_pos()
        button_texts = {
            "train_draw": "Train AI (with Visualization)",
            "train_no_draw": "Train AI (No Visualization - Faster)",
            "play_saved": "Play with Saved AI"
        }
        for name, rect in buttons.items():
            is_enabled = True if name != "play_saved" else model_exists
            draw_button(rect, button_texts[name], rect.collidepoint(mouse_pos), is_enabled)
        
        if not model_exists:
            status_msg = FONT_ARIAL_18.render('(No saved AI model found)', True, colors["text_disabled"])
            surface.blit(status_msg, (buttons["play_saved"].centerx-status_msg.get_width()//2, buttons["play_saved"].bottom+12))
        
        instr_surf = FONT_ARIAL_18.render("Press Q to quit | Click a button", True, colors["subtitle_color"])
        surface.blit(instr_surf, (S_WIDTH//2-instr_surf.get_width()//2, S_HEIGHT-40))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for name, rect in buttons.items():
                    if rect.collidepoint(mouse_pos) and (name != "play_saved" or model_exists):
                        return name
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q: return "quit"
        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__) if __file__ else os.getcwd() # Handles running from IDE vs script
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    best_ai_path = os.path.join(local_dir, 'best_tetris_ai.pkl')

    if not os.path.exists(config_path):
        print(f"Error: NEAT configuration file '{config_path}' not found.")
        print("Please create a 'config-feedforward.txt' file in the same directory as the script.")
        print("A sample config requires num_inputs = 4 for the current AI.")
        # Example:
        # [NEAT]
        # fitness_criterion     = max
        # fitness_threshold     = 100000 
        # pop_size              = 50
        # reset_on_extinction   = False
        # [DefaultGenome]
        # num_inputs              = 4
        # num_hidden              = 0 
        # num_outputs             = 1
        # initial_connection      = full_direct
        # feed_forward            = True
        # activation_default      = tanh
        # activation_mutate_rate  = 0.0
        # activation_options      = tanh
        # aggregation_default     = sum
        # aggregation_mutate_rate = 0.0
        # aggregation_options     = sum
        # bias_init_mean          = 0.0
        # bias_init_stdev         = 1.0
        # bias_max_value          = 30.0
        # bias_min_value          = -30.0
        # bias_mutate_power       = 0.5
        # bias_mutate_rate        = 0.7
        # bias_replace_rate       = 0.1
        # compatibility_disjoint_coefficient = 1.0
        # compatibility_weight_coefficient = 0.5
        # conn_add_prob           = 0.5
        # conn_delete_prob        = 0.5
        # enabled_default         = True
        # enabled_mutate_rate     = 0.01
        # node_add_prob           = 0.2
        # node_delete_prob        = 0.2
        # response_init_mean      = 1.0
        # response_init_stdev     = 0.0
        # response_max_value      = 30.0
        # response_min_value      = -30.0
        # response_mutate_power   = 0.0
        # response_mutate_rate    = 0.0
        # response_replace_rate   = 0.0
        # weight_init_mean        = 0.0
        # weight_init_stdev       = 1.0
        # weight_max_value        = 30
        # weight_min_value        = -30
        # weight_mutate_power     = 0.5
        # weight_mutate_rate      = 0.8
        # weight_replace_rate     = 0.1
        pygame.quit()
        exit()

    pygame.init()
    win_menu = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
    pygame.display.set_caption("NEAT Tetris AI - Main Menu")

    action = display_main_menu(win_menu, best_ai_path)
    
    pygame.display.quit() # Close menu window before starting game/training

    if action == "train_draw":
        print("Starting training with game drawing...")
        run_neat(config_path, draw_while_training=True)
        if os.path.exists(best_ai_path):
            print("Training finished. Playing with the best AI from this session.")
            play_with_ai(best_ai_path, config_path)
    elif action == "train_no_draw":
        print("Starting training without game drawing (faster)...")
        run_neat(config_path, draw_while_training=False)
        if os.path.exists(best_ai_path):
            print("Training finished. Playing with the best AI from this session.")
            play_with_ai(best_ai_path, config_path)
    elif action == "play_saved":
        if os.path.exists(best_ai_path):
            print("Playing with saved AI...")
            play_with_ai(best_ai_path, config_path)
        else:
            print("Saved AI model not found. Please train an AI first.")
    elif action == "quit":
        print("Exiting.")

    pygame.quit()
    import sys
    sys.exit()