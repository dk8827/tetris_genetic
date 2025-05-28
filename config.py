# config.py
import pygame

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

GRID_COLS = 10
GRID_ROWS = 20

# --- Colors ---
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (128, 0, 128)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE = (255, 165, 0)
COLOR_GREY = (128, 128, 128)
COLOR_LIGHT_GREY = (200, 200, 200)
COLOR_DARK_GREY = (50, 50, 50)
COLOR_BG = (15, 15, 25)


# --- Fonts ---
# Usage: FONT_MAP['comicsans_60']
FONT_MAP = {
    'comicsans_60': pygame.font.SysFont('comicsans', 60, bold=True),
    'comicsans_30': pygame.font.SysFont('comicsans', 30, bold=True),
    'comicsans_20': pygame.font.SysFont('comicsans', 20, bold=True),
    'comicsans_16': pygame.font.SysFont('comicsans', 16),
    'arial_64_bold': pygame.font.SysFont('arial', 64, bold=True),
    'arial_28_bold': pygame.font.SysFont('arial', 28, bold=True),
    'arial_20': pygame.font.SysFont('arial', 20),
    'arial_18': pygame.font.SysFont('arial', 18),
}

# --- Game Settings ---
INITIAL_FALL_SPEED = 0.30  # Seconds per drop for human player
LEVEL_UP_LINES = 10
FALL_SPEED_DECREMENT_PER_LEVEL = 0.03
MIN_FALL_SPEED = 0.05
SOFT_DROP_SCORE = 1
HARD_DROP_SCORE_MULTIPLIER = 2

# DAS (Delayed Auto Shift) and ARR (Auto Repeat Rate) for horizontal movement
H_MOVE_DELAY_FIRST_MS = 200  # ms before auto-repeat starts after first press
H_MOVE_INTERVAL_REPEAT_MS = 50   # ms between auto-repeats
# Soft drop speed
V_MOVE_INTERVAL_MS = 16 # ms for soft drop repeats

# NEAT Training settings
AI_GAME_FRAME_LIMIT_BASE = 1500
AI_GAME_FRAME_LIMIT_PER_LINE = 100
AI_FITNESS_LINE_CLEAR_BONUS = 10
AI_FITNESS_TETRIS_BONUS = 5 # Additional bonus for clearing 4 lines
AI_FITNESS_SURVIVAL_INCREMENT = 0.1
AI_FITNESS_GAME_OVER_PENALTY_FACTOR = 0.05 # Multiplied by sum of heights
AI_FITNESS_NO_MOVE_PENALTY = 10

# Score mapping for line clears
SCORE_MAP = {1: 40, 2: 100, 3: 300, 4: 1200}

# File Paths
HUMAN_HIGHSCORE_FILE = "human_highscore.txt"
BEST_AI_MODEL_FILE = 'best_tetris_ai.pkl'
NEAT_CONFIG_FILE = 'config-feedforward.txt'

# NN Visualization
NN_NODE_RADIUS = 7
NN_NODE_SPACING_Y_WITHIN_LAYER = 12
NN_LINE_THICKNESS_MULTIPLIER = 2.5
NN_PADDING = 15
NN_TITLE_COLOR = (220,220,220)
NN_BG_COLOR = (30,30,30)
NN_BORDER_COLOR = (80,80,80)