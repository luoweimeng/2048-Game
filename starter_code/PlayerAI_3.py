from random import randint
from BaseAI import BaseAI
from numpy import inf
from Grid_3 import Grid
from Displayer_3  import Displayer
import time
import math
import numpy as np

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)

actionDic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

class PlayerAI(BaseAI):
    def getMove(self, grid):
        maxChild, maxUtility = None, -inf
        prev_time = time.clock()
        depth_limit = 1

    # iterative deepen search
        while time.clock() - prev_time < 0.10:
            # print("depth_limit : ", depth_limit)
            initial_state = State(grid, 0, time.clock(), None, 0, depth_limit)
            child, utility = self.maximize(initial_state, -inf, inf)
            if utility > maxUtility:
                maxChild, maxUtility = child, utility
            depth_limit += 1
        return maxChild.prev_move

    # player
    def maximize(self, state, alpha, beta):
        if state.terminal_test():
            return None, state.eval()
        maxChild, maxUtility = None, -inf

        for child in state.get_children_max():
            _, utility = self.chance(child, alpha, beta)
            if utility > maxUtility:
                maxChild, maxUtility = child, utility

            if maxUtility >= beta:
                break

            if maxUtility > alpha:
                alpha = maxUtility
            # print(child.grid.map, utility)
            # print(child.max_value())
            # print(actionDic[child.prev_move])
        return maxChild, maxUtility

    # computer
    def minimize(self, state, alpha, beta):
        if state.terminal_test():
            return None, state.eval()
        minChild, minUtility = None, inf

        for child in state.get_children_min():
            _, utility = self.maximize(child, alpha, beta)

            if utility < minUtility:
                minChild, minUtility = child, utility

            if minUtility <= alpha:
                break

            if minUtility < beta:
                alpha = minUtility

        return minChild, minUtility

    def chance(self, state, alpha, beta):
        if state.terminal_test():
            return None, state.eval()
        chChild, chUtility = None, 0
        for child in state.get_children_chance():
            _, utility = self.minimize(child, alpha, beta)
            if child.tile_value == 2:
                chUtility += utility * 0.9
            elif child.tile_value == 4:
                chUtility += utility * 0.1

        return None, chUtility

class State:
    def __init__(self, grid, tile_value, start_time, move, depth, depth_limit):
        self.grid = grid
        self.tile_value = tile_value
        self.start_time = start_time
        self.prev_move = move
        self.depth = depth
        self.depth_limit = depth_limit

    def terminal_test(self):
        # print(time.clock() - self.start_time)
        return time.clock() - self.start_time > 0.10 or not self.grid.canMove() or self.depth > self.depth_limit

    def get_children_max(self):
        children = []
        # player
        for move in self.grid.getAvailableMoves():
            gridCopy = self.grid.clone()
            gridCopy.move(move)
            children.append(State(gridCopy, 0, self.start_time, move, self.depth + 1, self.depth_limit))
        return children

    def get_children_min(self):
        children = []
        for cell in self.grid.getAvailableCells():
            gridCopy = self.grid.clone()
            gridCopy.setCellValue(cell, self.tile_value)
            children.append(State(gridCopy, 0, self.start_time, None, self.depth + 1, self.depth_limit))
        return children

    def get_children_chance(self):
        children = []
        children.append(State(self.grid, 2, self.start_time, None, self.depth + 1, self.depth_limit))
        children.append(State(self.grid, 4, self.start_time, None, self.depth + 1, self.depth_limit))
        return children

    def eval(self):
        smooth_weight = 0.1
        mono_weight = 1.0
        empty_weight = 2.7
        max_weight = 5.0
        mean_weight = 5.0
        corner_weight = 10

        return (math.log(self.empty_cells()) if self.empty_cells() != 0 else 0) * empty_weight +\
            self.monotonicity() * mono_weight + self.max_value() * max_weight + \
            self.smoothness() * smooth_weight + self.max_value_at_corner() * corner_weight + self.mean_value() * mean_weight

    def mean_value(self):
        m = np.array(self.grid.map)
        m = m[m != 0]
        return math.log(m.mean(), 2)

    def max_value_at_corner(self):
        m = np.array(self.grid.map)
        index = m.argmax()
        if index in [0, 3, 12, 15]:
            return 1
        else:
            return 0

    def empty_cells(self):
        return (np.array(self.grid.map) == 0).sum()

    def max_value(self):
        m = np.max(np.array(self.grid.map))
        return math.log(m, 2)

    def monotonicity(self):
        # scores for all directions
        scores = [0, 0, 0, 0]
        # up/down direction
        for i in range(self.grid.size):
            current = 0
            next = current + 1
            while next < self.grid.size:
                while next < self.grid.size and self.grid.map[i][next] == 0:
                    next += 1
                if next >= 4:
                    next -= 1
                current_value = math.log(self.grid.map[i][current], 2) if self.grid.map[i][current] != 0 else 0
                next_value = math.log(self.grid.map[i][next], 2) if self.grid.map[i][next] != 0 else 0
                if current_value > next_value:
                    scores[0] += next_value - current_value
                elif next_value > current_value:
                    scores[1] += current_value - next_value
                current = next
                next += 1

        # left/right direction
        for j in range(self.grid.size):
            current = 0
            next = current + 1
            while next < self.grid.size:
                while next < self.grid.size and self.grid.map[next][j] == 0:
                    next += 1
                if next >= 4:
                    next -= 1
                current_value = math.log(self.grid.map[current][j], 2) if self.grid.map[current][j] != 0 else 0
                next_value = math.log(self.grid.map[next][j], 2) if self.grid.map[next][j] != 0 else 0
                if current_value > next_value:
                    scores[2] += next_value - current_value
                elif next_value > current_value:
                    scores[3] += current_value - next_value
                current = next
                next += 1
        return max((scores[0], scores[1])) + max((scores[2], scores[3]))

    def smoothness(self):
        smooth = 0
        for i in range(self.grid.size):
            for j in range(self.grid.size):
                if self.grid.map[i][j] != 0:
                    value = math.log(self.grid.map[i][j], 2)

                    next_right = self.find_next(i, j, True)
                    if next_right:
                        smooth -= math.fabs(math.log(next_right, 2) - value)
                    next_down = self.find_next(i, j, False)
                    if next_down:
                        smooth -= math.fabs(math.log(next_down, 2) - value)
        return smooth


    def find_next(self, i, j, right=True):
        if right:
            j = j + 1
            while j < self.grid.size and self.grid.map[i][j] == 0:
                j += 1
            return self.grid.map[i][j] if j < self.grid.size else None
        else:
            i = i + 1
            while i < self.grid.size and self.grid.map[i][j] == 0:
                i += 1
            return self.grid.map[i][j] if i < self.grid.size else None

if __name__ == '__main__':
    g = Grid()
    g.map = [[4, 0, 0, 0], [64, 16, 4, 0], [256, 128, 32, 8], [512, 256, 64, 16]]
    displayer = Displayer()
    displayer.display(g)
    s = State(g, 2, 0, None, 0, 0)
    print(s.smoothness())

    print(s.eval())
    testAI = PlayerAI()
    print(actionDic[testAI.getMove(g)])
    g.move(3)
    displayer.display(g)
    s1 = State(g, 2, 0, None, 0, 0)
    print(s1.eval())
    # for child in s.get_children_min():
    #     print(child.grid.map)