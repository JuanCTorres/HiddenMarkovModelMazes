"""
Author: Juan C. Torres
Date: November, 2017
"""
import numpy as np


class Maze:
    colors = ['r', 'g', 'b', 'y']
    WALL_STATE = -1
    WALL_CHAR = '#'
    DIRECTIONS = {'n': (1, 0), 's': (-1, 0), 'e': (0, 1), 'w': (0, -1)}

    def __init__(self, filename):
        self.matrix = self._read_matrix(filename)
        self.height = len(self.matrix)
        self.width = len(self.matrix[0])
        self.state_to_coord_dict = {}
        self.states = [list() for _ in range(self.height)]
        self.state_count = self._count_states()

    def __str__(self):
        return '\n'.join([str(r) for r in self.matrix])

    @staticmethod
    def _read_matrix(filename):
        with open(filename) as reader:  # split every line into a list of characters
            matrix = np.array([list(line.strip()) for line in reader.readlines()])
        return matrix

    def _count_states(self):
        """
        :return: the number of possible states (floor tiles, no walls) in the maze
        """
        count = 0
        for row in range(self.height):
            for col in range(self.width):
                if self.matrix[row, col] != self.WALL_CHAR:
                    state = count
                    count += 1
                else:
                    state = self.WALL_STATE
                    pass
                self.states[row].append(state)
                self.state_to_coord_dict[state] = (row, col)
        return count

    def is_floor(self, row, col):
        """
        Based on code from lab 2
        """
        if row < 0 or row >= self.height:
            return False
        if col < 0 or col >= self.width:
            return False

        return self.matrix[row][col] != self.WALL_CHAR


if __name__ == '__main__':
    # some simple testing code
    maze = Maze('../input/maze_walls.maz')
    print(maze.matrix)
    print(maze.states)
