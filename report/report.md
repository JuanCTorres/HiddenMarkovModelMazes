# Hidden Markov Models and first-order logic

Juan C. Torres

November 1, 2017

## Introduction

A robot is present in a maze and we want to know its location. The robot can move in four 
directions (N, S, E, W), and either move or hit a wall and remain in place. Each maze tile is painted in one of four 
colors: red, blue, green, or yellow. We want to estimate the probability that the robot is present at any one location. 

In other words, we want to generate a discrete probability distribution over all possible states the robot can have. Each state represents one possible location in the maze where the robot 
can be.

Our inputs are as follows:

* Knowledge of the maze layout
* A series of sensings by the robot. The robot has one camera pointing down, and thus generate a sensing for each step 
in time. The camera is not perfectly accurate, and as an 88% chance of sensing the correct color, and a 4% of sensing 
each of the remaining three incorrect colors. The robot has no other sensors, such as one to tell it whether it hit a 
wall or moved in any particular direction.

## Problem

We can formulate this problem as a first-order hidden Markov Model, where each state depends on the previous state. 
To do this, we apply the formulation of the forward algorithm as shown in Russell & Norvig, p. 579:

```
f{1: t + 1} = alpha * O_{t + 1} * tranpose(T) * f_{1 : t}, 
``` 
where:
- `t` represents the current step (one for each sensing), 
- `alpha` is a normalization factor, 
- `O` is a matrix containing the probabilities of seeing evidence `e` at step `t` given the state `i`:
 `P(e_t | X_t = i)` for each state `i`. For state `i`, such a number is located in the `i`th diagonal entry of the 
 matrix `O`. We have one such matrices for each color possible and choose which matrix to use based on the evidence seen
 at the current step in time.
-  `T` represents the transition matrix and contains the probability of moving from state `i` to
state `j` for all permutation of `i` and `j`. `T_{i, j}`.
-  `f` represents the probability, as seen by our model, of the robot being in each of the possible states. It is a 
1-dimensional vector. There is one of such vectors for every step in time, i.e., every evidence datum sensed by the 
robot.

## Implementation

I implemented a `Maze` class to contain the state of the maze: walls, actual colors of each tile, and so on.

I implemented the hidden Markov model in the `HMM` class. This class keeps track of a `Maze` object, as well as 
of the sensor inputs (the evidence), the `O` matrices for each color (`self.sensor_matrix_dict`), the transition matrix `T` (`self.transition_matrix`),
and the beliefs for all the steps in time `f` (`self.belief_list`).

### Generating the color matrices `O`:

For each color, we generate its sensor matrix as follows:

Loop over every possible state (the floor tiles) and append to an array the probability `P(e_t | X = i)`. 
Consider generating the sensor matrix for color `yellow`: for 
a state that is yellow, the probability of observing `yellow` is the probability that the camera will correctly 
capture that yellow: `0.88`. For state that is `red`, the probability that the camera will capture `yellow` (because this 
is the sensor matrix for `yellow`) is `0.04`. We then turn this into a diagonal matrix for convenience.

```python
def _build_color_matrix(self, color):
    prob_arr = []
    for row in range(self.maze.height):
        for col in range(self.maze.width):
            state_num = self.maze.states[row][col]
            if state_num != Maze.WALL_STATE:
                prob_arr.append(
                    self.CORRECT_COLOR_PROB if self.maze.matrix[row][col] == color else self.WRONG_COLOR_PROB
                )
    return np.diag(prob_arr)
```

### Generating the transition matrix `T`

We generate the transition matrix by looping over all possible states and getting, for each state, their neighbors and
the count of obstacles surrounding each state. If we know the number of neighbor states for state `i`, we can 
obtain the probability of transitioning from state `i` to each of its neighbor states (as well as to state `i`) as follows:
 
 A robot can move in one of four directions, so in a state
representing a tile with no walls, moving in each of these directions will have a `1/4` chance. For a state with
walls surrouding it, moving toward the walls will result in staying in place, so the chance of transtioning from `i` to `i`
goes up by `1/4` for every wall surrounding the state. Note this also implies that the probability of transitioning from state 
`i` to state `j` is either `1/4` or `0` for all `j` s.t. `j !=  i`. If `i == j`, this will be one of `{0, 1/4, 1/2, 3/4, 1}`.

I do this as follows in code:
```python
def _build_transition_matrix(self):
    transition = np.zeros([self.maze.state_count, self.maze.state_count])

    for row in range(self.maze.height):
        for col in range(self.maze.width):
            state_num = self.maze.states[row][col]
            if state_num != Maze.WALL_STATE:
                neighbor_list, wall_count = self._get_neighbor_cells(row, col)
                transition[state_num][state_num] = wall_count / self.MAX_NEIGHBOR_COUNT
                for neighbor in neighbor_list:
                    if neighbor != state_num:
                        transition[state_num][neighbor] = 1 / self.MAX_NEIGHBOR_COUNT
    return transition
```

The helper method `_get_neighbor_cells` is not particularly interesting: it checks whether the robot can move from a state
`i` to each of its neighbors (N, S, E, W) and returns the number of walls and a list of neighboring states for state `i`.


### Forward

The forward filtering code is mostly straightforward and follows the formulation given in Russell & Norvig, p. 579, as 
noted in the `problem` section before:

```
f{1: t + 1} = alpha * O_{t + 1} * tranpose(T) * f_{1 : t}
```

This can also be expressed as 

```
f{1: t} = alpha * O_{t} * tranpose(T) * f_{1 : t - 1},
```
which is a more convenient formulation to actually program this equation.

Note that I used the infix matrix multiplication operator `@`
introduced in python 3.5. For two matrices `A` and `B`, `A @ B` computes their matrix multiplication. Note that I loop 
over `1 to len(sensor_inputs) + 1` because I generate `len(sensor_inputs) + 1` probability distributions: one for each 
sensor input and one initial one to act as the base case of the recursion. The initial distribution is simply a uniform
distribution.  

```python
def forward(self):
    for i in range(1, len(self.sensor_inputs) + 1):  # step 0 is already done: a uniform disitribution
        # sensor matrix dict contains the diagonal matrices O, one for every color.
        res = self.sensor_matrix_dict[self.sensor_inputs[i - 1]] @ np.transpose(self.transition_matrix) @ \
              self.belief_list[i - 1]  # fyi, @ is an infix operator for matrix multiplication.
        alpha = 1 / sum(res)  # to normalize matrix
        self.belief_list.append(alpha * res)
    return self.belief_list
```

## Comparing the results with the actual position of the robot

Creating an `HMM` instance can be done either by passing to it a maze and a sequence of color sensings (the evidence) or
by passing simply a maze. In the former case, it is not possible to visualize the position of the robot; after all, our
only input is the sequence of sensings. In the latter case, we generate a list of random robot positions in the maze and simulate, for
each of those positions, capturing an image with the camera, with its 0.88 chance of getting the correct color and its 
0.04 chance of getting each of the wrong colors. For details of this, see `_simulate_robot_sensings`.

## Testing

To run the set of tests, run `main` in `HMM.py`. For each square of the maze, its actual color, the computed probability
the robot is there, and the robot location (if known) are shown. `#` characters represent walls.
 
Some highlights follow:
 
### maze_walls.maz

The initial state shows a uniform distribution over the possible states. Calculting this
distribution does **not** consume any evidence; the first piece of evidence, as noted previously, is consumed at step 1
```
TEST 1:
Sensor inputs: ['y', 'y', 'g', 'y', 'g', 'y', 'y', 'r', 'g', 'y', 'y', 'y', 'g', 'r', 'y', 'r'] 
step 0:
color observed: NONE
------------------------------------------------------------------
| R : 0.1250 | G : 0.1250 | # : 0.0000 | B : 0.1250 | Y : 0.1250 |
|            |            | ########## |            |            |
------------------------------------------------------------------
| Y : 0.1250 | Y : 0.1250 | G : 0.1250 | # : 0.0000 | R : 0.1250 |
|            |            |            | ########## |            |
------------------------------------------------------------------
```

The next step consumes the first piece of evidence:

```
step 1:
color observed: y
------------------------------------------------------------------
| R : 0.0141 | G : 0.0141 | # : 0.0000 | B : 0.0141 | Y : 0.3099 |
|            |            | ########## |            |            |
------------------------------------------------------------------
| Y : 0.3099 | Y : 0.3099 | G : 0.0141 | # : 0.0000 | R : 0.0141 |
| ROBOT      |            |            | ########## |            |
------------------------------------------------------------------
```

Final step:
```
step 16:
color observed: r
------------------------------------------------------------------
| R : 0.8582 | G : 0.0140 | # : 0.0000 | B : 0.0000 | Y : 0.0001 |
| ROBOT      |            | ########## |            |            |
------------------------------------------------------------------
| Y : 0.0747 | Y : 0.0422 | G : 0.0097 | # : 0.0000 | R : 0.0012 |
|            |            |            | ########## |            |
------------------------------------------------------------------
```

## maze_test_1.maz

This one is a large maze and does not fit in this report. Ascii result is shown below for the last 
step, however. **There might be some overflow on your screen if your markdown viewer 
does not create a scrollable view for this table!**

```
step 109:
color observed: y
----------------------------------------------------------------------------------------------------------------------
| R : 0.0005 | G : 0.0009 | # : 0.0000 | B : 0.0000 | Y : 0.0002 | R : 0.0000 | G : 0.0000 | # : 0.0000 | B : 0.0000 |
|            |            | ########## |            |            |            |            | ########## |            |
----------------------------------------------------------------------------------------------------------------------
| Y : 0.2986 | R : 0.0007 | G : 0.0015 | # : 0.0000 | R : 0.0000 | # : 0.0000 | # : 0.0000 | R : 0.0000 | G : 0.0000 |
|            |            |            | ########## |            | ########## | ########## |            |            |
----------------------------------------------------------------------------------------------------------------------
| B : 0.0135 | Y : 0.3235 | G : 0.0014 | # : 0.0000 | R : 0.0000 | R : 0.0000 | G : 0.0000 | # : 0.0000 | B : 0.0000 |
|            |            |            | ########## |            |            |            | ########## |            |
----------------------------------------------------------------------------------------------------------------------
| Y : 0.3128 | Y : 0.0383 | G : 0.0017 | # : 0.0000 | R : 0.0000 | # : 0.0000 | # : 0.0000 | R : 0.0000 | Y : 0.0000 |
| ROBOT      |            |            | ########## |            | ########## | ########## |            |            |
----------------------------------------------------------------------------------------------------------------------
| # : 0.0000 | # : 0.0000 | R : 0.0006 | B : 0.0005 | Y : 0.0052 | R : 0.0000 | G : 0.0000 | # : 0.0000 | B : 0.0000 |
| ########## | ########## |            |            |            |            |            | ########## |            |
----------------------------------------------------------------------------------------------------------------------
```

### maze_test_0.maz

Final probability distribution is shown below. **There might be some overflow on your screen if your markdown viewer 
does not create a scrollable view for this table!**
```
step 104:
color observed: r
----------------------------------------------------------------------------------------------------------------------
| R : 0.3626 | G : 0.0296 | # : 0.0000 | B : 0.0000 | Y : 0.0000 | R : 0.0000 | G : 0.0000 | # : 0.0000 | B : 0.0000 |
|            |            | ########## |            |            |            |            | ########## |            |
----------------------------------------------------------------------------------------------------------------------
| Y : 0.0014 | R : 0.4920 | G : 0.0142 | # : 0.0000 | R : 0.0000 | # : 0.0000 | # : 0.0000 | R : 0.0000 | G : 0.0000 |
|            | ROBOT      |            | ########## |            | ########## | ########## |            |            |
----------------------------------------------------------------------------------------------------------------------
| B : 0.0010 | Y : 0.0004 | G : 0.0105 | # : 0.0000 | R : 0.0001 | R : 0.0000 | G : 0.0000 | # : 0.0000 | B : 0.0000 |
|            |            |            | ########## |            |            |            | ########## |            |
----------------------------------------------------------------------------------------------------------------------
| Y : 0.0000 | Y : 0.0032 | G : 0.0035 | # : 0.0000 | R : 0.0006 | # : 0.0000 | # : 0.0000 | R : 0.0000 | Y : 0.0000 |
|            |            |            | ########## |            | ########## | ########## |            |            |
----------------------------------------------------------------------------------------------------------------------
| # : 0.0000 | # : 0.0000 | R : 0.0778 | B : 0.0006 | Y : 0.0002 | R : 0.0021 | G : 0.0002 | # : 0.0000 | B : 0.0000 |
| ########## | ########## |            |            |            |            |            | ########## |            |
----------------------------------------------------------------------------------------------------------------------
```

### Testing without knowledge of the robot's location

Changing the testing call to include a list to represent the evidence gathered the robot (so that no robot moves are 
generated) results in the robot not being shown in my tests:


```
TEST 4:
Sensor inputs: ['r', 'g', 'g'] 
step 0:
color observed: NONE
---------------------------
| R : 0.2500 | B : 0.2500 |
|            |            |
---------------------------
| G : 0.2500 | Y : 0.2500 |
|            |            |
---------------------------


step 1:
color observed: r
---------------------------
| R : 0.8800 | B : 0.0400 |
|            |            |
---------------------------
| G : 0.0400 | Y : 0.0400 |
|            |            |
---------------------------


step 2:
color observed: g
---------------------------
| R : 0.0736 | B : 0.0400 |
|            |            |
---------------------------
| G : 0.8800 | Y : 0.0064 |
|            |            |
---------------------------


step 3:
color observed: g
---------------------------
| R : 0.0250 | B : 0.0038 |
|            |            |
---------------------------
| G : 0.9493 | Y : 0.0219 |
|            |            |
---------------------------
```

## Extra credit

I noticed that the larger the maze, the larger the uncertainty for the actual location of the robot. When I generate the
sequence of locations of the robot, I scale up the number of observations the robot makes depending on the size of the 
maze so as to improve the results of the algorithm. I maintain some level of uncertainty by choosing at random the number
of steps within that scaled range. In doing so, the model became much more certain (higher probability) of knowing where
the robot was.

```python
def _simulate_robot_sensings(self):
    min_steps = max(self.maze.width, self.maze.height) ** 2
    max_steps = int(min_steps * 1.5)
    ...
    total_steps = random.randint(min_steps, max_steps)  # number of steps should depend on how big the maze is
    ...
```