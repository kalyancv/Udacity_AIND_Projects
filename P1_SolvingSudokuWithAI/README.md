# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: The program uses a combination of two main strategies (Constraint Propagation, and Search)
Constraint Propagation:

Strategy 1: Elimination: If a box has a value assigned, then none of the peers of this box can have this value.  So safely remove value from all other boxes in the peer.

Strategy 2:Only-choice: If there is only one box in a unit which would allow a certain digit, then that box must be assigned that digit.

Strategy3: Naked twins:  Within unit contains same two identical digit values, means no other boxes in the unit can't contains either of the values, then safely removed identical values from all other boxes in the unit.

Applying the same constraint as many times as possible until a solution is obtained, or the constraint can no longer be applied to refine the solution. Then use Search strategy.

Search strategy:
Pick a box with a minimal number of possible values. Try to solve each of the puzzles obtained by choosing each of these values, recursively.

# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: Add the diagonal units in the peers. And repeat same steps combination of two main strategies (Constraint Propagation, and Search).

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solution.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Submission
Before submitting your solution to a reviewer, you are required to submit your project to Udacity's Project Assistant, which will provide some initial feedback.  

The setup is simple.  If you have not installed the client tool already, then you may do so with the command `pip install udacity-pa`.  

To submit your code to the project assistant, run `udacity submit` from within the top-level directory of this project.  You will be prompted for a username and password.  If you login using google or facebook, visit [this link](https://project-assistant.udacity.com/auth_tokens/jwt_login for alternate login instructions.

This process will create a zipfile in your top-level directory named sudoku-<id>.zip.  This is the file that you should submit to the Udacity reviews system.

