# Simple grid, where there may be obstacles or not and the actions are:
# UP (0), RIGHT (1), DOWN (2), LEFT (3)
# (0, 0) is the top left corner

from MDP import MDP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


class GridWorldMDP(MDP):
    """
    A class representing a GridWorld Markov Decision Process (MDP).

    This class models a grid world environment where an agent navigates through a grid to reach a goal. The grid consists of various types of cells, such as normal cells, walls, a start position, and a goal position. The agent's task is to move from the start position to the goal while avoiding walls. The class supports both deterministic and stochastic dynamics for movement.

    Attributes:
    - NORMAL (int): Constant representing a normal position where the agent can be at.
    - WALL (int): Constant representing a wall blocking the agent's movement.
    - START (int): Constant representing the start position of the agent.
    - GOAL (int): Constant representing the goal position of the agent.
    - OFFSETS (dict): A dictionary mapping actions to directional offsets.
    - POSITIONS_CHAR (dict): A mapping of characters to cell types (normal, wall, start, goal).
    - POSITIONS (dict): A dictionary mapping cell types to a list of positions (coordinates) for that type.
    - CHAR_POSITIONS (dict): A reverse mapping of cell types to characters.
    - grid_size_x (int): The number of rows in the grid.
    - grid_size_y (int): The number of columns in the grid.
    - num_states (int): The total number of states (excluding terminal states).
    - start_pos (tuple[int, int]): The start position of the agent.
    - goal_pos (list[tuple[int, int]]): A list of goal positions.
    - agent_pos (tuple[int, int]): The current position of the agent.
    - num_actions (int): The number of possible actions the agent can take.
    - deterministic (bool): Whether the environment is deterministic or stochastic.
    - policy (list[int]): The optimal policy for each state.
    - policy_multiple_actions (list[list[int]]): The optimal policy for each state considering all the actions that maximize the reward.
    - V (list[float]): The value function for each state.
    - stats (Statistics): The statistics associated with the value iteration process.
    """

    NORMAL = 0
    WALL = 1
    START = 2
    GOAL = 3
    # ... --> can be extended in the future to acount for more wall types

    OFFSETS = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1)   # LEFT
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

    def __init__(self, grid_size: int = 3, start_pos: tuple[int, int] = (1, 1), map: list[str] = None, deterministic: bool = True):
        """
        Initializes the grid world based on the provided map or generates a simple grid. Also initializes matrices for state transitions (P) and rewards (R).

        Args:
        - grid_size (int, optional): The size of the grid (default is 3). The grid will have a size of grid_size x grid_size.
        - start_pos (tuple[int, int], optional): The starting position of the agent (default is (1, 1)).
        - map (list[str], optional): A custom map represented by a list of strings (default is None). If provided, the grid is loaded from this map.
        - deterministic (bool, optional): Whether the environment is deterministic (default is True).

        """
        assert grid_size > 0, "Grid size must be > 0"
        self.CHAR_POSITIONS = {v: k for k, v in self.POSITIONS_CHAR.items()}

        self.deterministic = deterministic

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
        super().__init__(self.num_states, num_terminal_states=1, num_actions=4)
        self.__generate_P()
        self.__generate_R()

    def __generate_simple_grid(self, grid_size: int):
        """
        Generates a simple grid with walls surrounding the grid and the goal positioned at the bottom-right corner.

        Args:
        - grid_size (int): The size of the grid (excluding the walls).
        """
        for i in range(grid_size):
            for j in range(grid_size):
                if (i == 0 or i == grid_size - 1) or (j == 0 or j == grid_size - 1):
                    self.POSITIONS[self.WALL].append((i, j))
                elif (i, j) != self.goal_pos[0]:
                    self.POSITIONS[self.NORMAL].append((i, j))
        self.POSITIONS[self.START].append(self.start_pos)
        self.POSITIONS[self.GOAL] = self.goal_pos

    def __load_from_map(self, map: list[str]):
        """
        Loads the grid world from a custom map represented as a list of strings. The map must have rows of equal length.

        Args:
        - map (list[str]): The grid map, where each character represents a type of cell.
        """
        assert all(len(row) == len(map[0]) for row in map), "Not all rows have the same length"

        for i, row in enumerate(map):
            for j, cell in enumerate(row):
                if cell == "S": self.POSITIONS[self.NORMAL].append((i, j))
                self.POSITIONS[self.POSITIONS_CHAR[cell]].append((i, j))

        self.start_pos = self.POSITIONS[self.START][0]
        self.goal_pos = self.POSITIONS[self.GOAL]
        self.num_states = len(self.POSITIONS[self.NORMAL]) + len(self.POSITIONS[self.GOAL])
        self.grid_size_x = len(map)
        self.grid_size_y = len(map[0])

    def __is_valid(self, pos: tuple[int, int]) -> bool:
        """
        Checks if a position is valid (i.e., not a wall).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is valid, False otherwise.
        """
        return pos not in self.POSITIONS[self.WALL]

    def __is_terminal(self, pos: tuple[int, int]) -> bool:
        """
        Checks if a position is a terminal state (i.e., a goal position).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is terminal, False otherwise.
        """
        return pos in self.goal_pos

    def __move(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool, bool]:
        """
        Computes the next position after performing an action, and returns whether the move is valid and whether the agent has reached a terminal state.

        Args:
        - pos (tuple[int, int]): The current position of the agent.
        - action (int): The action taken (0: up, 1: right, 2: down, 3: left).

        Returns:
        - tuple[tuple[int, int], bool, bool]: The next position, whether the move is valid, and whether the position is terminal.
        """
        x, y = pos
        dx, dy = self.OFFSETS[action]
        next_pos = (x + dx, y + dy)
        in_bounds = self.__is_valid(next_pos)
        if not in_bounds: next_pos = pos

        return next_pos, in_bounds, self.__is_terminal(next_pos)

    def compute_value_function(self):
        """
        Computes the value function using value iteration and extracts the optimal policy.
        """
        self.V, self.stats = self.value_iteration()
        
        self.policy = self.get_optimal_policy(self.V)
        self.policy_multiple_actions = self.get_optimal_policy(self.V, multiple_actions=True)

    def __generate_P(self):
        """
        Generates the transition probability matrix (P) for the grid world, based on the dynamics of the environment (deterministic or stochastic).
        """
        for state in range(self.num_non_terminal_states):
            for action in range(self.num_actions):
                next_state, _, _ = self.__move(self.POSITIONS[self.NORMAL][state], action)
                if next_state in self.POSITIONS[self.GOAL]:
                    next_state = self.POSITIONS[self.GOAL].index(next_state) + len(self.POSITIONS[self.NORMAL])
                else:
                    next_state = self.POSITIONS[self.NORMAL].index(next_state)

                if self.deterministic:
                    self.P[state, action, next_state] = 1
                else:
                    # Stochastic policy. With 70% take the correct action, with 30% take a random action
                    self.P[state, action, next_state] = 0.7
                    rand_action = np.random.choice([a for a in range(self.num_actions) if a != action])
                    next_state, _, _ = self.__move(self.POSITIONS[self.NORMAL][state], rand_action)
                    if next_state in self.POSITIONS[self.GOAL]:
                        next_state = self.POSITIONS[self.GOAL].index(next_state) + len(self.POSITIONS[self.NORMAL])
                    else:
                        next_state = self.POSITIONS[self.NORMAL].index(next_state)
                    self.P[state, action, next_state] += 0.3

    def __generate_R(self):
        """
        Generates the reward matrix (R) for the grid world, setting the default reward to -1 for all actions.
        """
        for state in range(self.num_non_terminal_states):
            self.R[state, :] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.int32)

    def print_grid(self):
        """
        Prints a textual representation of the grid world, showing the types of cells (normal, wall, start, goal).
        """
        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                type = [k for k, v in self.POSITIONS.items() if (i, j) in v][-1]
                print(self.CHAR_POSITIONS[type], end="")
            print()



class GridWorldPlotter:
    """
    A class responsible for plotting the GridWorld environment and its results, including the value function and policy.

    The `GridWorldPlotter` class is used to visualize the state of the grid world, including elements such as:
    - The layout of walls and goal positions
    - The value function of each state (if available)
    - The optimal policy as arrows indicating action directions

    Attributes:
    - NORMAL_COLOR (str): The color used for normal grid cells.
    - WALL_COLOR (str): The color used for walls in the grid world.
    - GOAL_COLOR (str): The color used for goal cells.
    - POLICY_COLOR (str): The color used for policy arrows.
    - gridworld (GridWorldMDP): An instance of the `GridWorldMDP` class that represents the grid world environment.
    - __figsize (tuple): The size of the figure for the plots.
    - __out_path (str): The output directory where the generated plots will be saved.
    """

    NORMAL_COLOR = "white"
    WALL_COLOR = "#383838"
    GOAL_COLOR = "#4EFF10"
    POLICY_COLOR = "#FF1010"

    def __init__(self, gridworld: GridWorldMDP, figsize: tuple[int, int] = (5, 5), name: str = None):
        """
        Initializes the GridWorldPlotter with the provided GridWorldMDP instance, figure size, and output directory.

        Args:
        - gridworld (GridWorldMDP): The GridWorldMDP instance that contains the grid world environment and its details.
        - figsize (tuple[int, int], optional): The size of the figure for plotting (default is (5, 5)).
        - name (str, optional): The name for the output directory where the plots will be saved.

        Creates the necessary directories for saving output if they do not exist.
        """
        self.gridworld = gridworld
        self.__figsize = figsize
        self.__out_path = os.path.join("assets", name)
        if not os.path.exists(self.__out_path): os.makedirs(self.__out_path)

    def plot_grid_world(self, savefig: bool = False, show_value_function: bool = False, policy: np.ndarray = None, multiple_actions: bool = False):
        """
        Plots the grid world environment, optionally showing the value function and policy.

        Args:
        - savefig (bool, optional): If True, the plot is saved to a file; otherwise, it is displayed (default is False).
        - show_value_function (bool, optional): If True, the value function of each state is displayed in the grid world (default is False).
        - policy (np.ndarray, optional): If exists, it uses the given policy. Otherwise, it uses the class policy

        If `show_value_function` is set to True, the grid will display a color map representing the value function. Otherwise, the grid will display the basic layout of the grid world, including walls and goal positions.
        The policy (optimal action) is visualized as arrows for each non-terminal state.
        """
        grid = np.full((self.gridworld.grid_size_x, self.gridworld.grid_size_y), self.gridworld.NORMAL)
        
        if policy is None:
            policy = self.gridworld.policy if not multiple_actions else self.gridworld.policy_multiple_actions
        else:
            print("WARNING: multiple actions in the policy found but `multiple_actions` parameter not set appropriately. Changing...")
            multiple_actions = isinstance(policy[0], list)

        for pos in self.gridworld.POSITIONS[self.gridworld.WALL]:
            grid[pos] = self.gridworld.WALL
        for pos in self.gridworld.POSITIONS[self.gridworld.GOAL]:
            grid[pos] = self.gridworld.GOAL

        fig, ax = plt.subplots(figsize=self.__figsize)

        if show_value_function:
            value_grid = np.zeros_like(grid, dtype=float)
            for idx, pos in enumerate(self.gridworld.POSITIONS[self.gridworld.NORMAL]):
                value_grid[pos] = self.gridworld.V[idx]
            
            # Walls should not be affected by the value function color
            for pos in self.gridworld.POSITIONS[self.gridworld.WALL]:
                value_grid[pos] = np.nan

            im = ax.imshow(value_grid, cmap="Blues", origin="upper")

            # Colorbar with the same height as the grid
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("Value Function", fontsize=12)

            # Put the walls as black
            for pos in self.gridworld.POSITIONS[self.gridworld.WALL]:
                ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color=self.WALL_COLOR))
            # Put the goal as green
            for pos in self.gridworld.POSITIONS[self.gridworld.GOAL]:
                ax.add_patch(plt.Rectangle((pos[1] - 0.5, pos[0] - 0.5), 1, 1, color=self.GOAL_COLOR))
        else:
            # Default color scheme for grid elements
            colormap = {
                self.gridworld.NORMAL: self.NORMAL_COLOR,
                self.gridworld.WALL: self.WALL_COLOR,
                self.gridworld.GOAL: self.GOAL_COLOR
            }
            color_list = [colormap[k] for k in sorted(colormap)]
            cmap = plt.matplotlib.colors.ListedColormap(color_list)
            ax.imshow(grid, cmap=cmap, origin="upper")

        # Add policy arrows
        for idx, pos in enumerate(self.gridworld.POSITIONS[self.gridworld.NORMAL]):
            actions = policy[idx] if multiple_actions else [policy[idx]]
            y, x = pos
            for action in actions:
                dx, dy = self.gridworld.OFFSETS[action]
                ax.quiver(
                    x, y, 0.35 * dy, 0.35 * dx, scale=1, scale_units="xy", angles="xy", 
                    width=0.005, color=self.POLICY_COLOR, headaxislength=3, headlength=3
                )

        # Adjust grid lines and labels
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color="gray", linestyle="-", linewidth=0.5)

        if savefig:
            plt.savefig(os.path.join(self.__out_path, f"{'deterministic' if self.gridworld.deterministic else 'stochastic'}_policy{'_value_function' if show_value_function else ''}"), dpi=300)
        else:
            plt.show()

    def plot_stats(self, savefig: bool=False):
        """
        Plots statistics for the grid world, including cumulative rewards over time.

        Args:
        - savefig (bool, optional): If True, the plot is saved to a file; otherwise, it is displayed (default is False).

        The statistics are based on the `GridWorldMDP` instance's `stats` attribute, which contains relevant data such as cumulative rewards.
        """
        self.gridworld.stats.print_statistics()
        fig = self.gridworld.stats.plot_cum_reward(deterministic=self.gridworld.deterministic, mdp=True)
        if savefig:
            fig.savefig(os.path.join(self.__out_path, f"{'deterministic' if self.gridworld.deterministic else 'stochastic'}_cum_reward.png"))
        else:
            plt.show()

        

