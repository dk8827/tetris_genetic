# main.py
import pygame
import os
import pickle
import sys # For sys.exit()

# Ensure project root is in python path if running scripts from subdirs (not usually needed for top-level)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_dir)


from config import S_WIDTH, S_HEIGHT, BEST_AI_MODEL_FILE, NEAT_CONFIG_FILE
from ui import MainMenu, NNVisualizer
from neat_trainer import NeatTrainer
from game_controller import HumanGameController, AIGameController
# Piece and Board are used by AIGameController indirectly, neat for AIPlayer/AIGameController
import neat # For neat.Config loading if genome is loaded without embedded config


def play_with_saved_ai(surface, model_path, neat_cfg_path_fallback):
    loaded_config = None
    genome = None
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2: # Expect (genome, config)
                genome, loaded_config = data
                if not isinstance(loaded_config, neat.Config): # Validate config type
                    print("Warning: Config loaded with genome is not a NEAT Config object. Trying fallback.")
                    loaded_config = None 
            else: # Old format, just genome
                genome = data
                print("Warning: Loaded model is in old format (genome only). Attempting to use fallback config.")
    except FileNotFoundError:
        print(f"Error: AI model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading AI model: {e}")
        return

    if not genome:
        print("No genome loaded.")
        return

    if not loaded_config: # If config wasn't bundled or was invalid
        if os.path.exists(neat_cfg_path_fallback):
            try:
                loaded_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                            neat_cfg_path_fallback)
            except Exception as e:
                print(f"Error loading NEAT config from {neat_cfg_path_fallback}: {e}")
                return
        else:
            print(f"Error: NEAT config required but not found (bundled or at {neat_cfg_path_fallback}).")
            return
    
    if not loaded_config:
        print("Error: NEAT configuration could not be loaded for the AI.")
        return

    pygame.display.set_caption("Tetris - AI Playing")
    net = neat.nn.FeedForwardNetwork.create(genome, loaded_config)
    
    # Reset global max score for AI playback session if it's tracked that way
    # Or pass a high score tracker to AIGameController if preferred
    # For now, AIGameController uses its own global for playback if no ai_stats are given
    try:
        delattr(AIGameController, '_GLOBAL_MAX_SCORE_TRACKER') # Reset if exists
    except AttributeError:
        pass 
    AIGameController._GLOBAL_MAX_SCORE_TRACKER = 0 # Explicitly set for playback

    ai_game = AIGameController(surface, net, loaded_config, draw_game=True)
    ai_game.run_for_playback()


if __name__ == '__main__':
    pygame.init() # Initialize all Pygame modules

    # Determine paths relative to this script file
    local_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    config_file_abs_path = os.path.join(local_dir, NEAT_CONFIG_FILE)
    best_ai_model_abs_path = os.path.join(local_dir, BEST_AI_MODEL_FILE)

    if not os.path.exists(config_file_abs_path):
        print(f"CRITICAL ERROR: NEAT configuration file '{config_file_abs_path}' not found.")
        print("Please ensure 'config-feedforward.txt' is in the same directory as main.py.")
        pygame.quit()
        sys.exit()

    main_surface = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
    pygame.display.set_caption("NEAT Tetris AI - Main Menu")
    
    menu = MainMenu(main_surface, best_ai_model_abs_path)
    action = menu.display()

    if action == "quit":
        pygame.quit()
        sys.exit()

    # Re-initialize display for the game/training if needed, or just change caption
    # Pygame modules like font, mixer are already init'd. Display needs to be managed.
    # If menu closes display, re-open. Here, we reuse `main_surface`.

    if action == "train_draw":
        print("Starting training with game drawing...")
        pygame.display.set_caption("NEAT Tetris - Training") # Initial caption
        trainer = NeatTrainer(config_file_abs_path)
        best_genome = trainer.run_training(generations=50, draw_while_training=True)
        if best_genome and os.path.exists(best_ai_model_abs_path):
            print("\nTraining finished. Playing with the best AI from this session...")
            play_with_saved_ai(main_surface, best_ai_model_abs_path, config_file_abs_path)
        elif not best_genome:
             print("\nTraining finished, but no best genome was determined or saved.")

    elif action == "train_no_draw":
        print("Starting training without game drawing (faster)...")
        # No Pygame window needed for display here, but trainer might still create one if not careful.
        # The trainer.eval_genomes and AIGameController now handle draw_game=False better.
        # We can close the menu window before starting non-drawing training if desired.
        # pygame.display.quit() # Example: close menu window
        # trainer_surface = None # Indicate no drawing surface
        # ... then re-init display if playing after.
        # For simplicity now, we assume non-drawing training doesn't use the display heavily.
        
        trainer = NeatTrainer(config_file_abs_path)
        best_genome = trainer.run_training(generations=50, draw_while_training=False)
        if best_genome and os.path.exists(best_ai_model_abs_path):
            print("\nTraining finished. Playing with the best AI from this session...")
            # Ensure display is active for playback
            main_surface = pygame.display.set_mode((S_WIDTH, S_HEIGHT)) # Re-activate if closed
            play_with_saved_ai(main_surface, best_ai_model_abs_path, config_file_abs_path)
        elif not best_genome:
             print("\nTraining finished, but no best genome was determined or saved.")


    elif action == "play_saved":
        if os.path.exists(best_ai_model_abs_path):
            print("Playing with saved AI...")
            play_with_saved_ai(main_surface, best_ai_model_abs_path, config_file_abs_path)
        else:
            # This case should ideally be handled by the menu disabling the button
            print("Saved AI model not found. Please train an AI first.")
            # Show message on screen? For now, console is fine.
            pygame.time.wait(2000)


    elif action == "human_play":
        print("Starting human player game...")
        pygame.display.set_caption("Tetris - Human Player")
        human_game = HumanGameController(main_surface)
        human_game.run()

    print("Exiting application.")
    pygame.quit()
    sys.exit()