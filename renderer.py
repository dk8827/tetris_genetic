# renderer.py
import pygame
from config import (S_WIDTH, S_HEIGHT, PLAY_WIDTH, PLAY_HEIGHT, BLOCK_SIZE,
                    TOP_LEFT_X, TOP_LEFT_Y, FONT_MAP, COLOR_BLACK, COLOR_WHITE,
                    COLOR_RED, COLOR_GREY, GRID_ROWS, GRID_COLS)
from shapes import SHAPE_LOOKUP # For next shape raw template

class Renderer:
    def __init__(self, surface):
        self.surface = surface

    def draw_play_area(self, board_visual_grid, current_piece=None):
        # Draw locked pieces from board_visual_grid
        for r in range(len(board_visual_grid)):
            for c in range(len(board_visual_grid[r])):
                if board_visual_grid[r][c] != COLOR_BLACK:
                    pygame.draw.rect(self.surface, board_visual_grid[r][c],
                                     (TOP_LEFT_X + c * BLOCK_SIZE, TOP_LEFT_Y + r * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 0)
        # Draw current piece
        if current_piece:
            for x_abs, y_abs in current_piece.get_absolute_coords():
                if y_abs >= 0: # Only draw parts of the piece that are on/below the top edge
                    pygame.draw.rect(self.surface, current_piece.color,
                                     (TOP_LEFT_X + x_abs * BLOCK_SIZE, TOP_LEFT_Y + y_abs * BLOCK_SIZE,
                                      BLOCK_SIZE, BLOCK_SIZE), 0)

        # Draw grid lines
        for i in range(GRID_ROWS + 1):
            pygame.draw.line(self.surface, COLOR_GREY,
                             (TOP_LEFT_X, TOP_LEFT_Y + i * BLOCK_SIZE),
                             (TOP_LEFT_X + PLAY_WIDTH, TOP_LEFT_Y + i * BLOCK_SIZE))
        for j in range(GRID_COLS + 1):
            pygame.draw.line(self.surface, COLOR_GREY,
                             (TOP_LEFT_X + j * BLOCK_SIZE, TOP_LEFT_Y),
                             (TOP_LEFT_X + j * BLOCK_SIZE, TOP_LEFT_Y + PLAY_HEIGHT))

        # Draw border
        pygame.draw.rect(self.surface, COLOR_RED,
                         (TOP_LEFT_X, TOP_LEFT_Y, PLAY_WIDTH, PLAY_HEIGHT), 5)

    def draw_next_shape(self, next_piece):
        if not next_piece: return

        label = FONT_MAP['comicsans_30'].render('Next Shape', 1, COLOR_WHITE)
        
        sx_base = TOP_LEFT_X + PLAY_WIDTH + 30
        sy_base = TOP_LEFT_Y + PLAY_HEIGHT / 2 - 150

        preview_block_size_factor = 0.7
        preview_block_size = BLOCK_SIZE * preview_block_size_factor
        
        # Use raw templates for simpler preview drawing
        shape_format_strings = SHAPE_LOOKUP.get_raw_template(next_piece.shape_idx, next_piece.rotation)
        
        # Calculate offset to center the 5x5 raw template display
        # Assuming the label is above, and we want to center the 5xN block grid
        max_template_width_chars = 5 # max(len(s) for s in shape_format_strings[0]) if shape_format_strings else 5
        max_template_height_chars = 5 # len(shape_format_strings) if shape_format_strings else 5
        
        preview_area_width = max_template_width_chars * preview_block_size
        
        # Center label above the shape preview area
        label_x = sx_base + (preview_area_width / 2) - (label.get_width() / 2)
        label_y = sy_base - label.get_height() - 10
        self.surface.blit(label, (label_x, label_y))

        shape_draw_sx = sx_base
        shape_draw_sy = sy_base

        for i, line in enumerate(shape_format_strings):
            for j, column_char in enumerate(line):
                if column_char == '0':
                    pygame.draw.rect(self.surface, next_piece.color,
                                     (shape_draw_sx + j * preview_block_size,
                                      shape_draw_sy + i * preview_block_size,
                                      preview_block_size, preview_block_size), 0)

    def draw_game_info(self, score, current_gen_max_score, training_overall_max_score, level=None, ai_stats=None):
        self.surface.fill(COLOR_BLACK) # Clear screen

        # Score
        score_label = FONT_MAP['comicsans_30'].render(f'Score: {score}', 1, COLOR_WHITE)
        sx_info_right = TOP_LEFT_X + PLAY_WIDTH + 30
        sy_info_start_right = TOP_LEFT_Y + PLAY_HEIGHT / 2 + 50 # Below next shape
        # Reposition Score to be on the left side for consistency
        info_x_base = TOP_LEFT_X - 250 # Base X for info on the left (adjust as needed)
        if info_x_base < 10: info_x_base = 10 # Ensure it's not off-screen

        current_y_offset = TOP_LEFT_Y + 20

        self.surface.blit(score_label, (info_x_base, current_y_offset))
        current_y_offset += score_label.get_height() + 10

        if ai_stats: # Training mode or AI with stats
            gen_max_label = FONT_MAP['comicsans_20'].render(f'Max Score (Gen): {current_gen_max_score}', 1, COLOR_WHITE)
            overall_max_label = FONT_MAP['comicsans_20'].render(f'Max Score (Train): {training_overall_max_score}', 1, COLOR_WHITE)
            
            self.surface.blit(gen_max_label, (info_x_base, current_y_offset))
            current_y_offset += gen_max_label.get_height() + 5
            self.surface.blit(overall_max_label, (info_x_base, current_y_offset))
            current_y_offset += overall_max_label.get_height() + 20 # More space before AI stats

            gen_label = FONT_MAP['comicsans_20'].render(f"Gen: {ai_stats['gen']}", 1, COLOR_WHITE)
            pop_label = FONT_MAP['comicsans_20'].render(f"Pop: {ai_stats['pop_size']}", 1, COLOR_WHITE)
            genome_label = FONT_MAP['comicsans_20'].render(f"Genome: {ai_stats['genome_id']}", 1, COLOR_WHITE)
            self.surface.blit(gen_label, (info_x_base, current_y_offset))
            current_y_offset += gen_label.get_height() + 5
            self.surface.blit(pop_label, (info_x_base, current_y_offset))
            current_y_offset += pop_label.get_height() + 5
            self.surface.blit(genome_label, (info_x_base, current_y_offset))

        elif level is not None: # Human player
            mode_label = FONT_MAP['comicsans_20'].render("Mode: Human Player", 1, COLOR_WHITE)
            highscore_label = FONT_MAP['comicsans_20'].render(f'High Score: {current_gen_max_score}', 1, COLOR_WHITE) # Use current_gen_max_score as it's the session high score for human
            level_label = FONT_MAP['comicsans_20'].render(f"Level: {level}", 1, COLOR_WHITE)
            
            self.surface.blit(mode_label, (info_x_base, current_y_offset))
            current_y_offset += mode_label.get_height() + 5
            self.surface.blit(highscore_label, (info_x_base, current_y_offset))
            current_y_offset += highscore_label.get_height() + 5
            self.surface.blit(level_label, (info_x_base, current_y_offset))
        
        else: # AI playing back (no training stats, no human level)
            # For AI Playback, current_gen_max_score and training_overall_max_score are the same (the global max for that playback)
            mode_label = FONT_MAP['comicsans_20'].render("Mode: AI Player", 1, COLOR_WHITE)
            max_score_playback_label = FONT_MAP['comicsans_20'].render(f'Max Score: {training_overall_max_score}', 1, COLOR_WHITE)

            self.surface.blit(mode_label, (info_x_base, current_y_offset))
            current_y_offset += mode_label.get_height() + 5
            self.surface.blit(max_score_playback_label, (info_x_base, current_y_offset))

    def draw_pause_message(self):
        pause_label = FONT_MAP['comicsans_60'].render("PAUSED", 1, COLOR_YELLOW)
        self.surface.blit(pause_label, (TOP_LEFT_X + PLAY_WIDTH / 2 - pause_label.get_width() / 2,
                                         TOP_LEFT_Y + PLAY_HEIGHT / 3 - pause_label.get_height() / 2))

    def draw_game_over_message(self):
        game_over_font = FONT_MAP['comicsans_60'] # Re-use comicsans 60 or define specific
        game_over_label = game_over_font.render("GAME OVER", 1, COLOR_RED)
        sub_text_font = FONT_MAP['comicsans_30']
        sub_text_label = sub_text_font.render("Press any key to continue", 1, COLOR_WHITE)

        self.surface.blit(game_over_label, (TOP_LEFT_X + PLAY_WIDTH / 2 - game_over_label.get_width() / 2,
                                             TOP_LEFT_Y + PLAY_HEIGHT / 3 - game_over_label.get_height() / 2))
        self.surface.blit(sub_text_label, (TOP_LEFT_X + PLAY_WIDTH / 2 - sub_text_label.get_width() / 2,
                                           TOP_LEFT_Y + PLAY_HEIGHT / 3 + game_over_label.get_height()))

    def update_display(self):
        pygame.display.update()