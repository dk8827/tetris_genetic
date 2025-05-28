# NEAT Tetris AI

## Description

This project implements a Tetris game where a NeuroEvolution of Augmenting Topologies (NEAT) algorithm learns to play. The AI is trained to make decisions based on features of the game state, such as the aggregate height of blocks, number of lines cleared by a move, holes in the board, and overall board bumpiness. The project uses Pygame for visualization and interaction, and features a modular codebase for improved organization and scalability.

## Features

*   **NEAT Algorithm**: AI learns to play Tetris through evolutionary computation.
*   **Pygame Interface**: Visual representation of the Tetris game, AI training process (optional), and the neural network of the best performing genome during training.
*   **Modular Design**: Code is organized into logical modules (e.g., `board.py`, `piece.py`, `game_controller.py`, `neat_trainer.py`, `ai_utils.py`).
*   **Training Modes**:
    *   Train with real-time visualization of games.
    *   Train without visualization for significantly faster computation.
*   **Play with Saved AI**: Load a previously trained and saved AI model (`best_tetris_ai.pkl`) to watch it play.
*   **Human vs. AI**: Option for human players to play Tetris.
*   **Customizable NEAT Configuration**: The `config-feedforward.txt` file allows tweaking of NEAT parameters.
*   **Main Menu**: User-friendly menu to select actions (train AI, play as AI, play as human).

## Setup and Installation

1.  **Python**: Ensure you have Python 3 installed (preferably 3.7 or newer).
2.  **Clone the Repository (if applicable)**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
3.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
4.  **Install Dependencies**:
    Install the required packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `pygame` and `neat-python`, among any other necessary dependencies.

## How to Run

Execute the main script from your terminal in the project's root directory:

```bash
python main.py
```

This will open a main menu with the following options:

*   **Train AI (with Visualization)**: Starts NEAT training with game rendering. Displays the best genome's neural network.
*   **Train AI (No Visualization - Faster)**: Starts NEAT training without game rendering for faster processing. Progress is shown in the console.
*   **Play with Saved AI**: Loads `best_tetris_ai.pkl` (if it exists) and lets you watch the AI play.
*   **Play Tetris (Human Controlled)**: Start a game of Tetris that you control.

After training, the best AI model is saved to `best_tetris_ai.pkl`. If training was initiated from the menu, the program may offer to play with this newly trained AI.

## NEAT Configuration

The `config-feedforward.txt` file defines parameters for the NEAT algorithm:

*   `pop_size`: Number of genomes per generation.
*   `fitness_threshold`: Target fitness score to stop evolution.
*   Input nodes (`num_inputs`): The network uses 4 inputs:
    1.  Aggregate height of all columns.
    2.  Number of lines cleared by the current move.
    3.  Number of holes in the board.
    4.  Board "bumpiness" (sum of height differences between adjacent columns).
*   Output nodes (`num_outputs`): The network has 1 output, representing the desirability of a potential move.
*   Other parameters include mutation rates, activation functions, speciation, etc.

Refer to the [NEAT-Python documentation](https://neat-python.readthedocs.io/en/latest/config_file.html) for details on all options.

## Controls

### Human Gameplay:
*   **Left/Right Arrow Keys**: Move piece horizontally.
*   **Down Arrow Key**: Soft drop piece.
*   **Up Arrow Key**: Hard drop piece.
*   **Space Bar**: Rotate piece.
*   **P**: Pause / Unpause game.
*   **Q**: Quit the current game (returns to menu or exits).

### AI Playback / Training with Visualization:
*   **Q**: Quit the visualization/playback.

### Main Menu:
*   **Mouse Click**: Select menu options.
*   **Q**: Quit the application.

## Notes

*   The AI evaluates moves by simulating piece placements in all valid positions and rotations. For each, it calculates board metrics, feeds them to its neural network, and chooses the move with the highest score.
*   Fitness in NEAT is primarily based on lines cleared, with considerations for game duration or score. 