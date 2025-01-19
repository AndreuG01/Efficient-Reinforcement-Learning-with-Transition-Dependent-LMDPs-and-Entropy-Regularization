# Simple grid, where there may be obstacles or not and the actions are:
# UP (0), RIGHT (1), DOWN (2), LEFT (3)
# (0, 0) is the top left corner

from MDP import MDP
import numpy as np


class GridWorldMDP(MDP):
    
    NORMAL = 0 # A regular position where the agent can be at
    WALL = 1 # A wall blocking the pass of the agent
    START = 2 # The starting position of the agent
    GOAL = 3 # A goal position of the agent
    # ... --> can be extended in the future to acount for more cell types
    
    OFFSETS = {
        0: (-1, 0), # UP
        1: (0, 1), # RIGHT
        2: (1, 0), # DOWN
        3: (0, -1) # LEFT
    }
    
    POSITIONS_CHAR = {
        " ": NORMAL,
        "#": WALL,
        "S": START,
        "G": GOAL
    }
    
    POSITIONS = {
        NORMAL: [],
        WALL: [],
        START: [],
        GOAL: []
    }
    
    def __init__(self, grid_size: int, start_pos: tuple[int, int] = (1, 1), map: list[str]=None):
        assert grid_size > 0, "Grid size must be > 0"
        self.CHAR_POSITIONS = {v: k for k, v in self.POSITIONS_CHAR.items()}
        
        if map is not None:
            self.__load_from_map(map)
        else:
            self.grid_size_x = grid_size + 2
            self.grid_size_y = grid_size + 2
            self.num_states = grid_size ** 2
            self.goal_pos = [(grid_size, grid_size)]
            self.start_pos = start_pos
            self.__generate_simple_grid(self.grid_size_x)
        
        self.agent_pos = self.start_pos
        super().__init__(self.num_states , num_terminal_states=1, num_actions=4)
        self.__generate_P()
        self.__generate_R()
    
    
    def __generate_simple_grid(self, grid_size: int):
        for i in range(grid_size):
            for j in range(grid_size):
                if (i == 0 or i == grid_size - 1) or (j == 0 or j == grid_size - 1): self.POSITIONS[self.WALL].append((i, j))
                elif (i, j) != self.goal_pos[0]: self.POSITIONS[self.NORMAL].append((i, j))
        self.POSITIONS[self.START].append(self.start_pos)
        self.POSITIONS[self.GOAL] = self.goal_pos
        
        print(self.POSITIONS[self.WALL])
        print(self.POSITIONS[self.NORMAL])
        print(self.POSITIONS[self.START])
        print(self.POSITIONS[self.GOAL][0])
            
    
    def __load_from_map(self, map: list[str]):
        
        assert all(len(row) == len(map[0]) for row in map), "Not all rows have the same length"
        
        for i, row in enumerate(map):
            for j, cell in enumerate(row):
                if cell == "S": self.POSITIONS[self.NORMAL].append((i, j))
                # if cell == "G": self.POSITIONS[self.NORMAL].append((i, j))
                self.POSITIONS[self.POSITIONS_CHAR[cell]].append((i, j))
            
        self.start_pos = self.POSITIONS[self.START][0]
        self.goal_pos = self.POSITIONS[self.GOAL]
        self.num_states = len(self.POSITIONS[self.NORMAL]) + len(self.POSITIONS[self.GOAL])
        self.grid_size_x = len(map)
        self.grid_size_y = len(map[0])
        
        print(f"Loaded grid with:\n\t{self.num_states} sates.\n\tStart position: {self.start_pos}.\n\tGoal positions: {self.goal_pos}.\n\tGrid size: {self.grid_size_x}x{self.grid_size_y}")
        print(len(self.POSITIONS[self.NORMAL]))
        print(self.POSITIONS[self.NORMAL])
        
    
    def __is_valid(self, pos: tuple[int, int]) -> bool:
        
        return pos not in self.POSITIONS[self.WALL]

    def __is_terminal(self, pos: tuple[int, int]) -> bool:
        # In the future this may be extended if there are more terminal positions
        return pos in self.goal_pos
    
    def __move(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool, bool]:
        """
        Returns: next_pos, is_valid, is_terminal
        """
        x, y = pos
        dx, dy = self.OFFSETS[action]
        next_pos = (x + dx, y + dy)
        in_bounds = self.__is_valid(next_pos)
        if not in_bounds: next_pos = pos
        
        return next_pos, in_bounds, self.__is_terminal(next_pos)
        

    
    def __generate_P(self):
        for state in range(self.num_non_terminal_states):
            for action in range(self.num_actions):
                next_state, _, _ = self.__move(self.POSITIONS[self.NORMAL][state], action)
                # print(f"sate: {state}, action: {action}, next_state: {next_state}")
                if next_state in self.POSITIONS[self.GOAL]:
                    next_state = self.POSITIONS[self.GOAL].index(next_state) + len(self.POSITIONS[self.NORMAL])
                else:
                    next_state = self.POSITIONS[self.NORMAL].index(next_state)
                self.P[state, action, next_state] = 1
        
    
    def __generate_R(self):
        for state in range(self.num_non_terminal_states):
            self.R[state, :] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.int32)

    
    def print_grid(self):
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                type = [k for k, v in self.POSITIONS.items() if (i, j) in v][-1]
                print(self.CHAR_POSITIONS[type], end="")
            print()


SIMPLE_TEST = [
    "####",
    "#S #",
    "# G#",
    "####"
]


LARGE_TEST = [
    "##########",
    "#    #  G#",
    "#    #   #",
    "#    #   #",
    "#    #   #",
    "#        #",
    "#S       #",
    "##########",
]


if __name__ == "__main__":
    grid_size = 2
    # mdp = GridWorldMDP(grid_size=grid_size, map=LARGE_TEST)
    mdp = GridWorldMDP(grid_size=grid_size)
    
    print("P")
    print(mdp.P.shape)
    # print(mdp.P[5, :, :])
    print("R")
    print(mdp.R)
    
    V = mdp.value_iteration()
    print(V)
    policy = mdp.get_optimal_policy(V)
    for i in range(len(policy)):
        print(f"State {i}: action {policy[i]}")
    print(policy)
    
    mdp.print_rewards()
    mdp.print_action_values(V)
    
    mdp.print_grid()
    
