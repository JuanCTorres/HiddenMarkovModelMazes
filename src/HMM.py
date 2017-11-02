"""
Author: Juan C. Torres
Date: November, 2017
"""
import random
import numpy as np
from Maze import Maze
from os import listdir


class HMM:
    CORRECT_COLOR_PROB = 0.88
    WRONG_COLOR_PROB = 0.04  #
    MAX_NEIGHBOR_COUNT = 4  # N, E, S, W

    def __init__(self, maze, sensor_data=None):
        self.maze = maze
        self.sensor_inputs = sensor_data if sensor_data is not None else self._simulate_robot_sensings()
        self.pos_is_known = sensor_data is None  # flag to know whether there's an outside observer with knowledge about the position
        self.sensor_matrix_dict = {color: self._build_color_matrix(color) for color in
                                   self.maze.colors}  # keys: 'r', 'g', ...
        self.transition_matrix = self._build_transition_matrix()
        self.belief_list = [self._generate_uniform_distribution(self.maze.state_count)]

    def forward(self):
        """
        Uses the recursive definition of filtering to generate probabilities that the robot is in each state:
        f_{1: t + 1} = alpha * O_{t + 1} * transpose(T) * f_{1 : t}. From Russell & Norvig, p. 579.
        :return:
        """
        for i in range(1, len(self.sensor_inputs) + 1):  # step 0 is already done: a uniform distribution
            # sensor matrix dict contains the diagonal matrices O, one for every color.
            res = self.sensor_matrix_dict[self.sensor_inputs[i - 1]] @ np.transpose(self.transition_matrix) @ \
                  self.belief_list[i - 1]  # fyi, @ is an infix operator for matrix multiplication.
            alpha = 1 / sum(res)  # to normalize matrix
            self.belief_list.append(alpha * res)
        return self.belief_list

    def _simulate_robot_sensings(self):
        """
        Simulates a robot starting at a random location and moving for a number of steps. For
        each steps, simulates sensing a color (0.88 chance of sensing right one, 0.04 of sensing each of the wrong ones)
        :return: list of color sensings
        """
        min_steps = max(self.maze.width, self.maze.height) ** 2
        # max_steps = max(self.maze.width, self.maze.height) *
        max_steps = int(min_steps * 1.5)
        # randint uses an inclusive range: 0..`count` - 1 adds to `count` states
        pos = self.maze.state_to_coord_dict[random.randint(0, self.maze.state_count - 1)]
        pos_list = [pos]

        assert self.maze.is_floor(*pos)  # make sure nothing went wrong
        total_steps = random.randint(min_steps, max_steps)  # number of steps should depend on how big the maze is
        for i in range(total_steps):
            neighbors, _ = self._get_neighbor_cells(*pos)
            pos = random.choice([self.maze.state_to_coord_dict[state] for state in neighbors])
            pos_list.append(pos)
        self.pos_list = pos_list
        return [self._simulate_sensor_reading(self.maze.matrix[i][j]) for i, j in pos_list]

    def _simulate_sensor_reading(self, color):
        """
        Simulates one sensor reading for `color`: 0.88 chance of getting the right one;
        otherwise uniform distribution between the wrong colors
        :param color: color to simulate the reading of
        :return: color sensed by the camera
        """
        num = random.uniform(0, 1)
        if num <= self.CORRECT_COLOR_PROB:
            return color
        else:
            choice = random.choice([c for c in self.maze.colors if c != color])
            return choice

    @staticmethod
    def _generate_uniform_distribution(state_count):
        return [1 / state_count] * state_count

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

    def _get_neighbor_cells(self, row, col):
        """
        :return: list of neighbors for (row, col) and count of number of neighbors
        """
        result = []
        possible_neighbors = [np.add([row, col], val) for val in Maze.DIRECTIONS.values()]
        for rr, cc in possible_neighbors:
            if self.maze.is_floor(rr, cc):
                result.append(curr_maze.states[rr][cc])
        return result, self.MAX_NEIGHBOR_COUNT - len(result)

    def display_result(self, res):
        """
        Displays a series of timesteps showing the belief states of the robot as well as the actual location
        of the robot, if available.

        :param res: Result containing modeled probabilities for every state in the maze for every time step
        :return: None
        """
        # 10-wide cells (+ 2 padding on each side for each),
        # 1 end-bar for each cell, 1 start bar for the leftmost cell
        cell_content_width = 10
        str_width = (cell_content_width * self.maze.width) + (2 * curr_maze.width) + (1 * self.maze.width) + 1
        print('Sensor inputs: %s ' % self.sensor_inputs)
        for i in range(len(res)):
            state = res[i]
            print('step %d:' % i)
            print('color observed: %s' % (self.sensor_inputs[i - 1] if i > 0 else 'NONE'))
            for row in range(len(self.maze.matrix)):
                print('-' * str_width)
                vals = [
                    self._get_probability_display_data(row, col, state) for col in
                    range(len(self.maze.matrix[row]))
                ]
                robot_pos = [
                    self._get_robot_display_data(row, col, i, cell_content_width) for col in
                    range(len(self.maze.matrix[row]))
                ]
                # Python does allow for pretty lovely one liners. This is just printing the actual
                # color of each square in this line and the probability the robot is there
                print('|%s|' % '|'.join([' %s : %.4f ' % (color.upper(), prob) for color, prob in vals]))
                # and this is printing the robot location, if known
                print('|%s|' % '|'.join([' %s ' % robot_str for robot_str in robot_pos]))
            print('-' * str_width)
            print('\n')

    def _get_probability_display_data(self, row, col, state_belief):
        """
        Get the color of this tile as well as the probability that the robot is there
        :param state_belief: the current beliefs of the robot about where it is located
        :return: (true color, P(robot_loc = (row, col))
        """
        true_color = self.maze.matrix[row][col]
        prob_curr_pos = state_belief[self.maze.states[row][col]] if true_color != self.maze.WALL_CHAR else 0
        return true_color, prob_curr_pos

    def _get_robot_display_data(self, row, col, state_ind, cell_width):
        empty_loc_str = ' ' * cell_width
        robot_repr = 'ROBOT'
        if self.maze.matrix[row][col] == self.maze.WALL_CHAR:
            return self.maze.WALL_CHAR * cell_width
        if not self.pos_is_known:
            return empty_loc_str
        robot_pos = self.pos_list[state_ind - 1] if state_ind > 0 else None  # no robot location known for 0-th state
        if robot_pos is None:
            return empty_loc_str
        return robot_repr.ljust(cell_width, ' ') if robot_pos == (row, col) else empty_loc_str


if __name__ == '__main__':
    input_dir = '../input/'
    test = 1
    for file in listdir(input_dir):
        print('TEST %d:' % test)
        curr_maze = Maze('%s%s' % (input_dir, file))
        h = HMM(curr_maze)
        res = h.forward()
        h.display_result(res)
        test += 1
