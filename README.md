# general-game-playing-tpg
A general game playing research project using Tangled Program Graphs (TPG), 
a genetic programming algorithm.

## Different Runs
- individuals: Each environment is trained on individually.

- 15-shrink-novir: Environments to train on selected randomly, with 9 in the training pool at a time, 
and 3 being swapped out per generation. Starts with 15 in the training pool, and slowly shrinks to 9.

- 15-shrink-vir: Environments to train on selected based on the environment fitness which is based on a 
host-parasite model, with 9 in the training pool at a time, and 3 being swapped out per generation. 
Starts with 15 in the training pool, and slowly shrinks to 9.

- 8-all-at-once: All environments being trained on every generation.

- 8-all-at-once-window-4: All environment being trained on from the ground up but only up to 4 at a time
selected randomly.

- 8-merge: All environments being trained on every generation, but continuing on from individually 
trained populations.

- 8-merge-window-4: Only up to 4 of the 8 environments being trained on every generation, selected 
randomly, but continuing on from individually trained populations.