# NEAT Tetris AI

## Description

This project implements a Tetris game where a NeuroEvolution of Augmenting Topologies (NEAT) algorithm learns to play. The AI is trained to make decisions based on features of the game state, such as aggregate height of blocks, number of cleared lines, holes, and board bumpiness. The project uses Pygame for visualization and interaction.

## Features

*   **NEAT Algorithm**: AI learns to play Tetris through evolutionary computation.
*   **Pygame Interface**: Visual representation of the Tetris game, AI training process, and the neural network of the best performing genome.
*   **Training Modes**:
    *   Train with real-time visualization of games.
    *   Train without visualization for faster computation.
*   **Play with Saved AI**: Load a previously trained and saved AI model (`best_tetris_ai.pkl`) to watch it play.
*   **Customizable NEAT Configuration**: The `config-feedforward.txt` file allows tweaking of NEAT parameters.
*   **Main Menu**: User-friendly menu to select actions (train, play).

## File Structure

*   `program.py`: The main Python script containing all the game logic, Pygame visualization, NEAT integration, and AI decision-making.
*   `config-feedforward.txt`: Configuration file for the NEAT algorithm. It defines parameters for the neural network structure, mutation rates, speciation, etc.
*   `best_tetris_ai.pkl`: This file is generated after training and stores the best-performing AI genome and its configuration. It's used by the "Play with Saved AI" feature.
*   `README.md`: This file.

## Setup and Installation

1.  **Python**: Ensure you have Python 3 installed (preferably 3.7 or newer).
2.  **Clone the Repository (if applicable)**:
    ```bash
    git clone <repository-url>
    cd neat-tetris-ai 
    ```
3.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
4.  **Install Dependencies**:
    The primary dependencies are `pygame` and `neat-python`.
    ```bash
    pip install pygame neat-python
    ```

## How to Run

Execute the main program script from your terminal:

```bash
python program.py
```

This will open a main menu with the following options:

*   **Train AI (with Visualization)**: Starts the NEAT training process and shows the games being played by each genome in the population. The neural network of the current best genome is also displayed.
*   **Train AI (No Visualization - Faster)**: Starts the NEAT training process without rendering the game. This significantly speeds up training. Progress is printed to the console.
*   **Play with Saved AI**: If a `best_tetris_ai.pkl` file exists (from a previous training session), this option loads the AI and lets you watch it play.

After training (either mode), if a new best AI is found, it will be saved to `best_tetris_ai.pkl`, and the program will automatically proceed to let you watch this newly trained AI play.

## NEAT Configuration

The file `config-feedforward.txt` contains all the parameters for the NEAT algorithm. You can modify this file to experiment with different settings:

*   `pop_size`: Number of genomes in each generation.
*   `num_inputs`: Number of input features to the neural network (currently 4: aggregate height, lines cleared, holes, bumpiness).
*   `num_outputs`: Number of output values from the neural network (currently 1: a score for the desirability of a move).
*   Mutation rates, activation functions, connection probabilities, and speciation thresholds.

Refer to the [NEAT-Python documentation](https://neat-python.readthedocs.io/en/latest/config_file.html) for a detailed explanation of all configuration options.

## Controls

*   **During AI Training Visualization / AI Playback**:
    *   `Q`: Quit the current game/visualization and exit the program.
*   **Main Menu**:
    *   Click on buttons to select options.
    *   `Q`: Quit the program.

## Notes

*   The AI evaluates potential moves by simulating the placement of the current piece in all possible rotations and horizontal positions. For each simulated placement, it calculates features of the resulting board state and feeds them into its neural network. The move corresponding to the highest output from the network is chosen.
*   Fitness for each genome is primarily based on the number of lines cleared, with additional small rewards for survival and penalties for game overs or undesirable board states (e.g., high stacks). 