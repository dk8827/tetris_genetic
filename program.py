import pygame
import random
import neat
import os
import pickle # To save and load the best genome

# --- Pygame Setup ---
pygame.font.init()
pygame.mixer.init() # For sound effects (optional)

# Screen dimensions
S_WIDTH = 800
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

def eval_genomes(genomes, config):
    global MAX_SCORE_SO_FAR, GENERATION_COUNT
    GENERATION_COUNT += 1
    win = pygame.display.set_mode((S_WIDTH, S_HEIGHT)) # Initialize once per generation if needed

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


            # --- Drawing ---
            # Only draw if you want to watch it train (slows down significantly)
            # For faster training, comment out drawing or draw only for best genome.
            # if GENERATION_COUNT % 5 == 0 and genome_id == genomes[0][0]: # Draw first genome every 5 gens
            if True: # Draw all
                draw_window(win, grid, score, MAX_SCORE_SO_FAR, GENERATION_COUNT, len(genomes), genome_id)
                draw_next_shape(next_piece, win)
                # Draw current piece (already handled by grid if locked, need to draw falling)
                temp_piece_draw = Piece(current_piece.x, current_piece.y, current_piece.shape)
                temp_piece_draw.rotation = current_piece.rotation
                temp_piece_draw.color = current_piece.color # Ensure color is set
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


def run_neat(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    p = neat.Population(config)

    # Add reporters for output
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5)) # Save checkpoints every 5 generations

    winner = p.run(eval_genomes, 50) # Run for 50 generations

    # Save the winner
    with open('best_tetris_ai.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    
    # Visualize the best genome playing (optional)
    # You'd need a separate function similar to eval_genomes but just for one genome
    # and without fitness updates, just playing.


def play_with_ai(genome_path, config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

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


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    
    # To train:
    # run_neat(config_path)

    # To play with a saved AI:
    best_ai_path = os.path.join(local_dir, 'best_tetris_ai.pkl')
    if os.path.exists(best_ai_path):
         play_with_ai(best_ai_path, config_path)
    else:
        print("No saved AI found. Training a new one...")
        run_neat(config_path)
        print("Training finished. Now playing with the best AI.")
        play_with_ai(best_ai_path, config_path)