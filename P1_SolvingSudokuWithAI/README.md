# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: The program uses a combination of two main strategies (Constraint Propagation, and Search)
Constraint Propagation:

Strategy 1: Elimination: If a box has a value assigned, then none of the peers of this box can have this value.  So safely remove value from all other boxes in the peer.

Strategy 2:Only-choice: If there is only one box in a unit which would allow a certain digit, then that box must be assigned that digit.

Strategy3: Naked twins:  Within unit contains same two identical digit values, means no other boxes in the unit can't contains either of the values, then safely removed identical values from all other boxes in the unit.

Applying the same constraint as many times as possible until a solution is obtained, or the constraint can no longer be applied to refine the solution. Then use Search strategy.

Search strategy:
Pick a box with a minimal number of possible values. Try to solve each of the puzzles obtained by choosing each of these values, recursively.

# (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: Add the diagonal units in the peers. And repeat same steps combination of two main strategies (Constraint Propagation, and Search).

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

