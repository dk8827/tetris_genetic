# input_handler.py
import pygame
from game_state import GameState
from config import (H_MOVE_DELAY_FIRST_MS, H_MOVE_INTERVAL_REPEAT_MS,
                    V_MOVE_INTERVAL_MS)

class InputHandler:
    def __init__(self):
        self.h_key_down_time = {pygame.K_LEFT: 0, pygame.K_RIGHT: 0}
        self.time_since_last_h_input_ms = 0
        self.time_since_last_v_input_ms = 0

    def reset_das_arr_timers(self):
        self.time_since_last_h_input_ms = 0
        self.time_since_last_v_input_ms = 0
        self.h_key_down_time[pygame.K_LEFT] = 0
        self.h_key_down_time[pygame.K_RIGHT] = 0


    def handle_input(self, game_controller, delta_time_ms):
        """
        Processes Pygame events and continuous key presses.
        Calls methods on the game_controller to perform actions.
        Returns True if the game should quit, False otherwise.
        """
        if game_controller.game_state == GameState.GAME_OVER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return True
                if event.type == pygame.KEYDOWN: return True # Any key to exit game over
            return False

        # Update timers for continuous movement
        self.time_since_last_h_input_ms += delta_time_ms
        self.time_since_last_v_input_ms += delta_time_ms

        # Event-based input (single presses, pause)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    game_controller.toggle_pause()
                    continue
                
                if game_controller.game_state == GameState.PAUSED:
                    continue # Ignore game inputs if paused

                if event.key == pygame.K_LEFT:
                    game_controller.move_piece_horizontally(-1)
                    self.h_key_down_time[pygame.K_LEFT] = pygame.time.get_ticks()
                    self.h_key_down_time[pygame.K_RIGHT] = 0
                    self.time_since_last_h_input_ms = -H_MOVE_DELAY_FIRST_MS # Trigger DAS delay
                elif event.key == pygame.K_RIGHT:
                    game_controller.move_piece_horizontally(1)
                    self.h_key_down_time[pygame.K_RIGHT] = pygame.time.get_ticks()
                    self.h_key_down_time[pygame.K_LEFT] = 0
                    self.time_since_last_h_input_ms = -H_MOVE_DELAY_FIRST_MS
                elif event.key == pygame.K_DOWN: # Soft drop initial press
                    game_controller.soft_drop()
                    self.time_since_last_v_input_ms = 0
                elif event.key == pygame.K_SPACE: # Rotate
                    game_controller.rotate_piece()
                elif event.key == pygame.K_UP: # Hard Drop
                    game_controller.hard_drop()
                elif event.key == pygame.K_q:
                    return True
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT: self.h_key_down_time[pygame.K_LEFT] = 0
                elif event.key == pygame.K_RIGHT: self.h_key_down_time[pygame.K_RIGHT] = 0

        if game_controller.game_state == GameState.PAUSED:
            return False

        # Continuous input (auto-repeat for horizontal, fast soft drop)
        current_ticks = pygame.time.get_ticks()
        keys_pressed_now = pygame.key.get_pressed()

        # Horizontal auto-repeat
        if self.h_key_down_time[pygame.K_LEFT] > 0 and \
           current_ticks - self.h_key_down_time[pygame.K_LEFT] > H_MOVE_DELAY_FIRST_MS:
            if self.time_since_last_h_input_ms >= H_MOVE_INTERVAL_REPEAT_MS:
                self.time_since_last_h_input_ms = 0
                game_controller.move_piece_horizontally(-1)
        
        if self.h_key_down_time[pygame.K_RIGHT] > 0 and \
           current_ticks - self.h_key_down_time[pygame.K_RIGHT] > H_MOVE_DELAY_FIRST_MS:
            if self.time_since_last_h_input_ms >= H_MOVE_INTERVAL_REPEAT_MS:
                self.time_since_last_h_input_ms = 0
                game_controller.move_piece_horizontally(1)
        
        # Vertical soft drop auto-repeat
        if keys_pressed_now[pygame.K_DOWN] and self.time_since_last_v_input_ms >= V_MOVE_INTERVAL_MS:
            self.time_since_last_v_input_ms = 0
            game_controller.soft_drop()
            
        return False # Game continues