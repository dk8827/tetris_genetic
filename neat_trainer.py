# neat_trainer.py
import pygame # For window creation if drawing
import neat
import pickle
import os

from game_controller import AIGameController, GameState
from ui import NNVisualizer
from config import S_WIDTH, S_HEIGHT, BEST_AI_MODEL_FILE

class NeatTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                         config_path)
        self.population = neat.Population(self.neat_config)
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        # self.population.add_reporter(neat.Checkpointer(5)) # Optional

        self.current_generation = 0
        self.best_genome_overall = None
        self.highest_fitness_overall = -float('inf')
        self.max_score_training_overall_actual = 0 # Tracks the max score seen in the entire training session
        
        self.pygame_surface_train = None
        self.nn_visualizer_train = None


    def eval_genomes(self, genomes, config, draw_while_training=True):
        self.current_generation += 1
        if draw_while_training and not self.pygame_surface_train:
            self.pygame_surface_train = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
            pygame.display.set_caption(f"NEAT Tetris - Training Gen: {self.current_generation}")
            self.nn_visualizer_train = NNVisualizer(self.pygame_surface_train)
        elif draw_while_training and self.pygame_surface_train: # Update caption for existing window
             pygame.display.set_caption(f"NEAT Tetris - Training Gen: {self.current_generation}")


        AIGameController.reset_max_score_current_generation() # Reset for this generation's max score display

        for genome_id_tuple, genome in genomes:
            # The genome_id from NEAT can be a tuple if parallel, or just an int.
            # For display, we just need a unique identifier.
            actual_genome_id_for_display = genome.key 

            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            ai_stats_info = {
                'gen': self.current_generation,
                'pop_size': len(genomes),
                'genome_id': actual_genome_id_for_display
            }

            # Create an AIGameController instance for this genome
            # If not drawing, surface can be None or a dummy surface.
            # For now, AIGameController expects a surface.
            # If draw_while_training is False, we still need a surface for AIGameController,
            # but it won't be actively drawn to or updated on screen.
            # A cleaner way would be for AIGameController to not require a surface if not drawing.
            # For now, pass the training surface if it exists, or a dummy one.
            current_surface_for_game = self.pygame_surface_train if draw_while_training else pygame.Surface((S_WIDTH, S_HEIGHT))


            game_sim = AIGameController(current_surface_for_game, net, config, 
                                        draw_game=draw_while_training, 
                                        ai_stats_info=ai_stats_info,
                                        max_score_training_overall=self.max_score_training_overall_actual)
            
            genome.fitness = game_sim.run_for_evaluation(
                nn_visualizer=self.nn_visualizer_train if draw_while_training else None,
                current_genome_for_viz=genome if draw_while_training else None, # Pass current genome
                best_genome_for_viz=self.best_genome_overall if draw_while_training else None, 
                neat_config_for_viz=self.neat_config if draw_while_training else None
            )

            # Update the overall max score for the entire training session
            if game_sim.score > self.max_score_training_overall_actual:
                self.max_score_training_overall_actual = game_sim.score

            if genome.fitness is not None and genome.fitness > self.highest_fitness_overall:
                self.highest_fitness_overall = genome.fitness
                self.best_genome_overall = genome
        
        # Optional: If drawing, update display one last time per generation with overall best NN
        if draw_while_training and self.pygame_surface_train and self.best_genome_overall:
            # This part is tricky as eval_genomes is per genome.
            # A final "generation summary" draw could happen here, or rely on individual game draws.
            # For now, individual game draws already show the best_genome_overall's NN.
            pass


    def run_training(self, generations=50, draw_while_training=True):
        self.current_generation = 0 # Reset for a new training run
        self.best_genome_overall = None
        self.highest_fitness_overall = -float('inf')
        self.max_score_training_overall_actual = 0 # Reset for new training session
        AIGameController.reset_max_score_current_generation() # Reset gen max at start of full training


        winner_genome = self.population.run(
            lambda genomes, config: self.eval_genomes(genomes, config, draw_while_training),
            generations
        )

        # Save the best genome found during training (winner_genome is best of last gen)
        # self.best_genome_overall should hold the best across all generations
        best_to_save = self.best_genome_overall if self.best_genome_overall else winner_genome

        if best_to_save:
            # Save the best genome and its config
            with open(BEST_AI_MODEL_FILE, 'wb') as f:
                pickle.dump((best_to_save, self.neat_config), f)
            print(f'\nBest genome saved to {BEST_AI_MODEL_FILE}')
            print(f'{best_to_save}')
        else:
            print("No winning genome found or saved.")
            
        if self.pygame_surface_train: # Close the training window if it was opened
            # pygame.display.quit() # This quits the entire display module.
            # We might want to keep it if we immediately play with AI.
            # For now, let's assume main.py handles display re-init if needed.
            pass # Let main.py handle final pygame.quit()

        return best_to_save # Return the absolute best genome found