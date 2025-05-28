# game_controller.py
import pygame
import os
from enum import Enum, auto

from board import Board
from piece import Piece
from renderer import Renderer
from input_handler import InputHandler # For HumanGameController
from ai_utils import AIPlayer # For AIGameController
from config import (S_WIDTH, PLAY_HEIGHT, TOP_LEFT_X, PLAY_WIDTH, BLOCK_SIZE, TOP_LEFT_Y,INITIAL_FALL_SPEED, LEVEL_UP_LINES, FALL_SPEED_DECREMENT_PER_LEVEL, MIN_FALL_SPEED,
                    SOFT_DROP_SCORE, HARD_DROP_SCORE_MULTIPLIER, SCORE_MAP, HUMAN_HIGHSCORE_FILE,
                    AI_GAME_FRAME_LIMIT_BASE, AI_GAME_FRAME_LIMIT_PER_LINE)
from game_state import GameState # Import GameState from the new file

class BaseGameController:
    def __init__(self, surface):
        self.surface = surface
        self.renderer = Renderer(surface)
        self.board = Board()
        self.current_piece = Piece.get_new_piece()
        self.next_piece = Piece.get_new_piece()
        self.score = 0
        self.game_state = GameState.PLAYING
        self.clock = pygame.time.Clock()
        self.max_score_so_far_session = 0 # For displaying high score during game

    def _handle_piece_locked(self):
        self.board.lock_piece(self.current_piece)
        cleared_lines = self.board.clear_rows()
        
        if cleared_lines > 0:
            self._update_score_and_level(cleared_lines)

        self.current_piece = self.next_piece
        self.next_piece = Piece.get_new_piece()

        if self.board.check_lost() or not self.board.is_valid_space(self.current_piece):
            self.game_state = GameState.GAME_OVER
            # For AI, fitness might be penalized here or by caller.
            # For Human, save high score etc.
            if isinstance(self, HumanGameController):
                self._save_highscore()
            return True # Game is over
        return False # Game continues

    def _update_score_and_level(self, cleared_lines):
        # To be implemented by subclasses (Human vs AI might score differently or have levels)
        pass

    def render_game(self):
        # To be implemented by subclasses (info displayed might differ)
        pass

    def run(self):
        """Main game loop to be implemented by subclasses."""
        raise NotImplementedError


class HumanGameController(BaseGameController):
    def __init__(self, surface):
        super().__init__(surface)
        self.level = 1
        self.lines_cleared_total = 0
        self.fall_speed = INITIAL_FALL_SPEED
        self.fall_time_ms = 0
        self.input_handler = InputHandler()
        self.session_high_score = self._load_highscore()
        self.max_score_so_far_session = self.session_high_score # Initialize with loaded highscore

    def _load_highscore(self):
        try:
            if os.path.exists(HUMAN_HIGHSCORE_FILE):
                with open(HUMAN_HIGHSCORE_FILE, "r") as f:
                    return int(f.read())
        except (FileNotFoundError, ValueError):
            pass
        return 0

    def _save_highscore(self):
        if self.score > self.session_high_score:
            self.session_high_score = self.score
        try:
            with open(HUMAN_HIGHSCORE_FILE, "w") as f:
                f.write(str(self.session_high_score))
        except Exception as e:
            print(f"Could not save high score: {e}")

    def _update_score_and_level(self, cleared_lines):
        self.lines_cleared_total += cleared_lines
        self.score += SCORE_MAP.get(cleared_lines, 0) * self.level
        
        if self.score > self.max_score_so_far_session:
             self.max_score_so_far_session = self.score

        new_level = (self.lines_cleared_total // LEVEL_UP_LINES) + 1
        if new_level > self.level:
            self.level = new_level
            self.fall_speed = max(MIN_FALL_SPEED, INITIAL_FALL_SPEED - (self.level - 1) * FALL_SPEED_DECREMENT_PER_LEVEL)

    def toggle_pause(self):
        if self.game_state == GameState.PLAYING:
            self.game_state = GameState.PAUSED
        elif self.game_state == GameState.PAUSED:
            self.game_state = GameState.PLAYING
            self.input_handler.reset_das_arr_timers() # Reset timers on unpause

    def move_piece_horizontally(self, dx):
        self.current_piece.move(dx, 0)
        if not self.board.is_valid_space(self.current_piece):
            self.current_piece.move(-dx, 0) # Revert

    def soft_drop(self):
        self.current_piece.move(0, 1)
        if not self.board.is_valid_space(self.current_piece):
            self.current_piece.move(0, -1) # Revert; piece will lock on next auto-fall
        else:
            self.score += SOFT_DROP_SCORE
            self.fall_time_ms = 0 # Reset auto-fall timer

    def rotate_piece(self):
        original_rotation = self.current_piece.rotation
        original_x = self.current_piece.x
        self.current_piece.rotate()
        
        if not self.board.is_valid_space(self.current_piece):
            # Basic wall kick attempt (SRS is more complex)
            # Try moving right by 1
            self.current_piece.move(1,0)
            if not self.board.is_valid_space(self.current_piece):
                self.current_piece.move(-2,0) # Try moving left by 1 from original (total -2 from current)
                if not self.board.is_valid_space(self.current_piece):
                    self.current_piece.move(1,0) # Back to original x
                    self.current_piece.rotation = original_rotation # Revert rotation
            # No need to else: if a kick worked, it's now in a valid space.

    def hard_drop(self):
        original_y = self.current_piece.y
        while self.board.is_valid_space(self.current_piece):
            self.current_piece.move(0, 1)
        self.current_piece.move(0, -1) # Back to last valid position
        
        dropped_dist = self.current_piece.y - original_y
        self.score += dropped_dist * HARD_DROP_SCORE_MULTIPLIER
        
        self._handle_piece_locked()
        self.fall_time_ms = 0 # Reset fall timer as piece is now locked

    def render_game(self):
        visual_grid = self.board.create_visual_grid()
        self.renderer.draw_game_info(self.score, self.max_score_so_far_session, level=self.level)
        self.renderer.draw_play_area(visual_grid, self.current_piece if self.game_state != GameState.GAME_OVER else None)
        self.renderer.draw_next_shape(self.next_piece)

        if self.game_state == GameState.PAUSED:
            self.renderer.draw_pause_message()
        elif self.game_state == GameState.GAME_OVER:
            self.renderer.draw_game_over_message()
        
        self.renderer.update_display()

    def run(self):
        running = True
        while running:
            delta_time_ms = self.clock.tick(60) # Target 60 FPS

            if self.input_handler.handle_input(self, delta_time_ms):
                running = False # Quit signal from input handler
                continue

            if self.game_state == GameState.PLAYING:
                self.fall_time_ms += delta_time_ms
                if self.fall_time_ms / 1000.0 >= self.fall_speed:
                    self.fall_time_ms = 0
                    self.current_piece.move(0, 1)
                    if not self.board.is_valid_space(self.current_piece):
                        self.current_piece.move(0, -1) # Revert
                        if self._handle_piece_locked(): # Piece locked, game might be over
                           pass # game_state is now GAME_OVER if true
            
            self.render_game()
        
        self._save_highscore() # Ensure score is saved on exit


class AIGameController(BaseGameController):
    def __init__(self, surface, neat_network, neat_config, draw_game=True, ai_stats_info=None):
        super().__init__(surface)
        self.ai_player = AIPlayer(neat_network, neat_config)
        self.draw_game = draw_game
        self.genome_fitness = 0
        self.total_lines_cleared_genome = 0
        self.game_frames = 0
        self.ai_stats_info = ai_stats_info # Dict: {'gen': G, 'pop_size': P, 'genome_id': ID} for display
        self.max_score_so_far_training_session = 0 # Used if drawing during training

        # For AI playback, we might need a global max score reference if not training
        global _GLOBAL_MAX_SCORE_TRACKER # Ugly, better to pass it in if needed for playback
        if not self.ai_stats_info and '_GLOBAL_MAX_SCORE_TRACKER' not in globals():
            _GLOBAL_MAX_SCORE_TRACKER = 0
        
        if not self.ai_stats_info: # If playing back (not training)
            self.max_score_so_far_session = _GLOBAL_MAX_SCORE_TRACKER
        else: # If training
            # This might be better managed by NeatTrainer class
            self.max_score_so_far_session = AIGameController.get_global_max_score()


    # Static variable to track max score across all AI genomes in a training session
    # This is a bit of a workaround for the global MAX_SCORE_SO_FAR
    _ai_training_session_max_score = 0

    @staticmethod
    def reset_global_max_score():
        AIGameController._ai_training_session_max_score = 0
    
    @staticmethod
    def get_global_max_score():
        return AIGameController._ai_training_session_max_score

    def _update_score_and_level(self, cleared_lines): # Level not used by AI, but score for display
        self.total_lines_cleared_genome += cleared_lines
        self.score += SCORE_MAP.get(cleared_lines, 0)
        
        # Fitness calculation specific to AI training
        self.genome_fitness += cleared_lines * 10 # Example: config.AI_FITNESS_LINE_CLEAR_BONUS
        if cleared_lines == 4:
            self.genome_fitness += 5 # Example: config.AI_FITNESS_TETRIS_BONUS

        if self.ai_stats_info: # If in training mode
            if self.score > AIGameController._ai_training_session_max_score:
                AIGameController._ai_training_session_max_score = self.score
            self.max_score_so_far_session = AIGameController._ai_training_session_max_score
        else: # Playback mode
            global _GLOBAL_MAX_SCORE_TRACKER
            if self.score > _GLOBAL_MAX_SCORE_TRACKER:
                _GLOBAL_MAX_SCORE_TRACKER = self.score
            self.max_score_so_far_session = _GLOBAL_MAX_SCORE_TRACKER


    def render_game(self, nn_visualizer=None, best_genome_for_viz=None, neat_config_for_viz=None):
        if not self.draw_game: return

        visual_grid = self.board.create_visual_grid()
        self.renderer.draw_game_info(self.score, self.max_score_so_far_session, ai_stats=self.ai_stats_info)
        self.renderer.draw_play_area(visual_grid, self.current_piece if self.game_state != GameState.GAME_OVER else None)
        self.renderer.draw_next_shape(self.next_piece)

        if nn_visualizer and best_genome_for_viz and neat_config_for_viz:
             nn_area_x = TOP_LEFT_X + PLAY_WIDTH + 30 + 5*BLOCK_SIZE + 20 # From original
             nn_area_y = TOP_LEFT_Y + 50
             nn_area_width = S_WIDTH - nn_area_x - 20
             nn_area_height = PLAY_HEIGHT - 200
             if nn_area_width > 50 and nn_area_height > 50:
                nn_visualizer.draw(best_genome_for_viz, neat_config_for_viz,
                                   nn_area_x, nn_area_y, nn_area_width, nn_area_height)

        if self.game_state == GameState.GAME_OVER: # AI doesn't pause
            self.renderer.draw_game_over_message() # Or some other indication for AI
        
        self.renderer.update_display()

    def run_one_turn_ai(self):
        """Runs one decision cycle for the AI. Returns True if game over."""
        best_move = self.ai_player.choose_best_move(self.board, self.current_piece)

        if best_move:
            target_x, target_rot, target_y = best_move
            self.current_piece.x = target_x
            self.current_piece.rotation = target_rot
            # The AI already determined the final Y, so we can place it there directly
            # For correctness with board state, simulate the fall if target_y isn't just piece.y
            # The AIPlayer's choose_best_move already factors in the drop.
            # So, we just set the piece's y to the determined final y.
            # This assumes target_y is the y-coordinate of the anchor *after* dropping.
            self.current_piece.y = target_y

            if self._handle_piece_locked(): # Piece locked, game might be over
                # AI specific game over fitness penalty
                if self.ai_stats_info: # Training
                    final_heights = self.board.get_column_heights()
                    self.genome_fitness -= sum(final_heights) * 0.05 # Example: config.AI_FITNESS_GAME_OVER_PENALTY_FACTOR
                return True # Game over
        else:
            # AI found no valid move, game over
            if self.ai_stats_info: # Training
                self.genome_fitness -= 10 # Example: config.AI_FITNESS_NO_MOVE_PENALTY
            self.game_state = GameState.GAME_OVER
            return True # Game over
        
        if self.ai_stats_info: # Training
            self.genome_fitness += 0.1 # Example: config.AI_FITNESS_SURVIVAL_INCREMENT
        return False # Game continues

    def run_for_evaluation(self, nn_visualizer=None, best_genome_for_viz=None, neat_config_for_viz=None):
        """Runs the game for AI evaluation until game over or frame limit."""
        self.game_state = GameState.PLAYING
        
        while self.game_state == GameState.PLAYING:
            self.game_frames += 1

            if self.draw_game:
                self.clock.tick() # Run as fast as possible if drawing, or set a sensible tick
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); import sys; sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q: pygame.quit(); import sys; sys.exit()
            
            if self.run_one_turn_ai(): # AI makes a move, returns True if game over
                break # Exit loop if game over

            if self.draw_game:
                self.render_game(nn_visualizer, best_genome_for_viz, neat_config_for_viz)
            
            # Frame limit for AI training to prevent infinite games
            if self.game_frames > AI_GAME_FRAME_LIMIT_BASE + (self.total_lines_cleared_genome * AI_GAME_FRAME_LIMIT_PER_LINE):
                self.game_state = GameState.GAME_OVER # Consider this a timeout
                break
        
        return self.genome_fitness


    def run_for_playback(self):
        """Runs the game with AI control for demonstration, until game over."""
        self.game_state = GameState.PLAYING
        running = True
        while running and self.game_state == GameState.PLAYING:
            self.clock.tick(10) # Playback speed (e.g., 10 moves per second)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q: running = False
            
            if not running: break

            if self.run_one_turn_ai(): # AI makes a move
                self.render_game() # Render final game over state
                pygame.time.wait(2000) # Show game over screen for a bit
                break 
            
            self.render_game()
        
        # If AI playback ends, keep window open until user quits
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN: running = False # Any key to exit after game over
            self.clock.tick(15) # Keep Pygame responsive