from enum import Enum, auto

class GameState(Enum):
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto() 