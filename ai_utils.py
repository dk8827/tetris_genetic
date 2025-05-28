# ai_utils.py
import neat # For type hinting if needed, not direct use in AIEvaluator
from piece import Piece
from board import Board # For type hinting and creating hypothetical boards
from config import GRID_COLS, GRID_ROWS, COLOR_BLACK # For simulation parameters

class AIEvaluator:
    """
    Simulates a piece placement and evaluates the resulting board state.
    This class does not interact with NEAT networks directly.
    """
    def __init__(self, board_to_simulate_on: Board, piece_to_place: Piece):
        self.original_board = board_to_simulate_on
        self.piece_template = piece_to_place # This is the piece type, not placed yet

    def simulate_placement(self, target_x_anchor: int, target_rotation_idx: int):
        """
        Simulates placing the piece with a given rotation at a given column,
        dropping it down as far as possible.
        Returns:
            - hypothetical_board: A new Board object representing the state after placement.
            - final_y_anchor: The y-coordinate where the piece locked.
            - lines_cleared_by_this_move: Number of lines cleared.
            - True if placement is valid and doesn't immediately lose, False otherwise.
        """
        sim_board = self.original_board.clone()
        
        # Create a piece instance for simulation
        # Start high to ensure it can drop into place
        sim_piece = Piece.from_properties(target_x_anchor, 0, 
                                          self.piece_template.shape_idx, 
                                          target_rotation_idx)

        # Drop the piece
        # First, check if the initial spawn position (at y=0, or slightly above) is valid
        # Adjust initial y if needed, or just ensure valid_space handles y<0 correctly
        if not sim_board.is_valid_space(sim_piece):
             # If it's not valid even at the top (e.g. due to immediate collision or out of bounds x),
             # this move (column/rotation combo) is impossible from the start.
             # However, typical Tetris allows spawning partially off-screen above.
             # Let's assume for AI, we try to place it and drop.
             # A more robust check would be to find highest valid Y, but the drop loop handles this.
             pass # Allow it to proceed, drop loop will handle validity

        initial_y = -2 # Start piece a bit above the screen to simulate drop
        sim_piece.y = initial_y
        
        # Find landing position
        while sim_board.is_valid_space(sim_piece):
            sim_piece.y += 1
        sim_piece.y -= 1 # Last valid position

        # If piece is still entirely above the board after "dropping", it's an invalid move path.
        # This can happen if the piece is too wide for a narrow passage high up.
        all_blocks_above_board = True
        for _, y_abs in sim_piece.get_absolute_coords():
            if y_abs >=0:
                all_blocks_above_board = False
                break
        if all_blocks_above_board and sim_piece.y < 0: # If it never entered the board
            # This move is effectively impossible or leads to no change relevant to board metrics
            # Or handle as a very bad move.
            return None, -1, 0, False


        # Lock the piece on the hypothetical board
        sim_board.lock_piece(sim_piece)

        # Check for game over *after* locking
        if sim_board.check_lost():
            return sim_board, sim_piece.y, 0, False # Lost state

        lines_cleared = sim_board.clear_rows()

        return sim_board, sim_piece.y, lines_cleared, True

    @staticmethod
    def get_board_metrics(board_state: Board):
        """Calculates metrics for a given board state."""
        agg_height = board_state.get_aggregate_height()
        holes = board_state.get_num_holes()
        bumpiness = board_state.get_bumpiness()
        return agg_height, holes, bumpiness


class AIPlayer:
    """
    Uses a NEAT network and an AIEvaluator to determine the best move.
    """
    def __init__(self, neat_network: neat.nn.FeedForwardNetwork, config=None): # config might be needed for input normalization ranges
        self.net = neat_network
        self.config = config # NEAT config, not game config

    def choose_best_move(self, current_board: Board, current_piece_template: Piece):
        """
        Determines the best move (rotation, x_position, final_y_position) for the AI.
        Returns: (final_x_anchor, final_rotation_idx, final_y_anchor) or None.
        """
        best_move_score = -float('inf')
        best_move_details = None  # (final_x_anchor, final_rotation_idx, final_y_anchor)

        evaluator = AIEvaluator(current_board, current_piece_template)
        
        num_rotations = current_piece_template._num_rotations

        for r_idx in range(num_rotations):
            # Create a temporary piece to get its width for column iteration range
            temp_piece_for_bounds = Piece.from_properties(0,0, current_piece_template.shape_idx, r_idx)
            current_rel_coords = temp_piece_for_bounds.get_relative_coords()
            if not current_rel_coords: continue

            min_dx = min(coord[0] for coord in current_rel_coords) # Smallest x offset from anchor
            max_dx = max(coord[0] for coord in current_rel_coords) # Largest x offset from anchor

            # Iterate through possible anchor columns
            # Anchor x is column index. Piece extends from x+min_dx to x+max_dx
            # So, 0 <= x+min_dx  => x >= -min_dx
            # And x+max_dx < GRID_COLS => x < GRID_COLS - max_dx
            for c_anchor_x in range(-min_dx, GRID_COLS - max_dx):
                hypothetical_board, final_y, lines_cleared, is_valid_and_not_lost = \
                    evaluator.simulate_placement(c_anchor_x, r_idx)

                if not is_valid_and_not_lost or hypothetical_board is None:
                    # This move leads to a loss or is impossible, skip or heavily penalize
                    # For now, we only consider non-losing moves.
                    continue

                agg_h, holes, bump = AIEvaluator.get_board_metrics(hypothetical_board)

                # Normalize inputs for the network (example normalization, might need tuning)
                # These max values are empirical guesses.
                # config-feedforward usually defines input nodes from -X to X (e.g. -0 to -3 for inputs)
                # So the NN expects inputs, not their names.
                # Order of inputs must match neat_config.txt input_nodes!
                # Default inputs often are: agg_height, lines_cleared, holes, bumpiness
                nn_inputs = (
                    agg_h / (GRID_ROWS * GRID_COLS),    # Max possible agg_height is 200 (20*10)
                    lines_cleared / 4.0,           # Max 4 lines cleared
                    holes / (GRID_ROWS * GRID_COLS), # Max possible holes (rough estimate)
                    bump / (GRID_ROWS * (GRID_COLS-1)) # Max bumpiness (rough estimate)
                )

                output = self.net.activate(nn_inputs)
                current_move_score = output[0] # Assuming single output neuron for move rating

                if current_move_score > best_move_score:
                    best_move_score = current_move_score
                    best_move_details = (c_anchor_x, r_idx, final_y)
        
        return best_move_details