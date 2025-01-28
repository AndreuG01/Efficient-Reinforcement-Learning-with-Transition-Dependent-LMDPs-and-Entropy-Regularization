# Simple grid, where there may be obstacles or not and the actions are:
# UP (0), RIGHT (1), DOWN (2), LEFT (3)
# (0, 0) is the top left corner

from models.MDP import MDP
from models.LMDP import LMDP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from .grid import CustomGrid, CellType
from utils.state import State
 
                
class GridWorldMDP(MDP):
    """
    A class representing a GridWorld Markov Decision Process (MDP).

    This class models a grid world environment where an agent navigates through a grid to reach a goal. The grid consists of various types of cells, such as normal cells, walls, a start position, and a goal position. The agent's task is to move from the start position to the goal while avoiding walls. The class supports both deterministic and stochastic dynamics for movement.

    Attributes:
    - OFFSETS (dict): A dictionary mapping actions to directional offsets.
    - size_x (int): The number of rows in the grid.
    - size_y (int): The number of columns in the grid.
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

    OFFSETS = {
        0: (0, -1),  # UP
        1: (1, 0),   # RIGHT
        2: (0, 1),   # DOWN
        3: (-1, 0)   # LEFT
    }

    def __init__(self, grid_size: int = 3, map: list[str] = None, deterministic: bool = True):
        """
        Initializes the grid world based on the provided map or generates a simple grid. Also initializes matrices for state transitions (P) and rewards (R).

        Args:
        - grid_size (int, optional): The size of the grid (default is 3). The grid will have a size of grid_size x grid_size.
        - start_pos (tuple[int, int], optional): The starting position of the agent (default is (1, 1)).
        - map (list[str], optional): A custom map represented by a list of strings (default is None). If provided, the grid is loaded from this map.
        - deterministic (bool, optional): Whether the environment is deterministic (default is True).

        """
        assert grid_size > 0, "Grid size must be > 0"
        

        self.deterministic = deterministic
        
        self.grid = CustomGrid(map=map, grid_size=grid_size)
        self.num_states = self.grid.get_num_states()
        
        
        start_pos = self.grid.start_pos
        
        super().__init__(
            self.num_states,
            num_terminal_states=self.grid.get_num_terminal_states(),
            allowed_actions=[i for i in range(len(self.OFFSETS))],
            s0=self.grid.positions[CellType.NORMAL].index(State(*start_pos))
        )
        
        self.generate_P(self.grid.positions, self.move, self.grid)
        self._generate_R()
    
    
    def move(self, state: State, action: int) -> tuple[State, bool, bool]:
        """
        Computes the next position after performing an action, and returns whether the move is valid and whether the agent has reached a terminal state.

        Args:
        - pos (tuple[int, int]): The current position of the agent.
        - action (int): The action taken (0: up, 1: right, 2: down, 3: left).

        Returns:
        - tuple[tuple[int, int], bool, bool]: The next position, whether the move is valid, and whether the position is terminal.
        """
        y = state.y
        x = state.x
        dy, dx = self.OFFSETS[action]
        next_state = State(y + dy, x + dx)
        
        in_bounds = self.grid.is_valid(next_state)
        if not in_bounds: next_state = state
        
        # if next_pos in self.grid.positions[self.CLIFF]: next_pos = self.start_pos

        return next_state, in_bounds, self.grid.is_terminal(next_state)


    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the grid world, setting the default reward to -1 for all actions.
        """
        pos = self.grid.positions
        for j in range(self.grid.size_x):
            for i in range(self.grid.size_y):
                tmp_state = State(i, j)
                if tmp_state in pos[CellType.NORMAL]:
                    self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.int32)
                if tmp_state in pos[CellType.CLIFF]:
                    self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-10, dtype=np.int32)



class GridWorldLMDP(LMDP):
    
    OFFSETS = {
        0: (0, -1),  # UP
        1: (1, 0),   # RIGHT
        2: (0, 1),   # DOWN
        3: (-1, 0)   # LEFT
    }
    
    def __init__(self, grid_size: int = 3, map: list[str] = None) -> None:
        self.grid = CustomGrid(map=map, grid_size=grid_size)
        self.num_sates = self.grid.get_num_states()
        
        start_pos = self.grid.start_pos
        
        super().__init__(
            self.num_sates,
            num_terminal_states=self.grid.get_num_terminal_states(),
            s0=self.grid.positions[CellType.NORMAL].index(State(*start_pos))
        )
        
        self.allowed_actions = [i for i in range(len(self.OFFSETS))] # It is not that the LMDP has actions, but to determine the transition probabilities, we need to know how the agent moves through the environment
        
        self.generate_P(self.grid.positions, self.move, self.grid, self.allowed_actions)
        self._generate_R()
    
    
    # TODO: the same as GridWorldMDP. Perhaps could be unified somehow.
    def move(self, state: State, action: int) -> tuple[State, bool, bool]:
        """
        Computes the next position after performing an action, and returns whether the move is valid and whether the agent has reached a terminal state.

        Args:
        - pos (tuple[int, int]): The current position of the agent.
        - action (int): The action taken (0: up, 1: right, 2: down, 3: left).

        Returns:
        - tuple[tuple[int, int], bool, bool]: The next position, whether the move is valid, and whether the position is terminal.
        """
        y = state.y
        x = state.x
        dy, dx = self.OFFSETS[action]
        next_state = State(y + dy, x + dx)
        
        in_bounds = self.grid.is_valid(next_state)
        if not in_bounds: next_state = state
        
        # if next_pos in self.grid.positions[self.CLIFF]: next_pos = self.start_pos

        return next_state, in_bounds, self.grid.is_terminal(next_state)
    
    
    # TODO: same as GridWorldMDP. Perhaps could be unified somehow.
    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the grid world, setting the default reward to -1 for all actions.
        """
        pos = self.grid.positions
        for j in range(self.grid.size_x):
            for i in range(self.grid.size_y):
                tmp_state = State(i, j)
                if tmp_state in pos[CellType.NORMAL]:
                    self.R[pos[CellType.NORMAL].index(tmp_state)] = -1
                if tmp_state in pos[CellType.CLIFF]:
                    self.R[pos[CellType.NORMAL].index(tmp_state)] = -10


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
    CLIFF_COLOR = "#3F043C"
    POLICY_COLOR = "#FF1010"

    def __init__(self, gridworld: GridWorldMDP, figsize: tuple[int, int] = (5, 5), name: str = ""):
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

    def plot_grid_world(self, savefig: bool = False, save_title: str = None, show_value_function: bool = False, policy: np.ndarray = None, multiple_actions: bool = False):
        """
        Plots the grid world environment, optionally showing the value function and policy.

        Args:
        - savefig (bool, optional): If True, the plot is saved to a file; otherwise, it is displayed (default is False).
        - show_value_function (bool, optional): If True, the value function of each state is displayed in the grid world (default is False).
        - policy (np.ndarray, optional): If exists, it uses the given policy. Otherwise, it uses the class policy

        If `show_value_function` is set to True, the grid will display a color map representing the value function. Otherwise, the grid will display the basic layout of the grid world, including walls and goal positions.
        The policy (optimal action) is visualized as arrows for each non-terminal state.
        """
        grid = np.full((self.gridworld.grid.size_x, self.gridworld.grid.size_y), CellType.NORMAL)
        grid_positions = self.gridworld.grid.positions
        
        if policy is None:
            policy = self.gridworld.policy if not multiple_actions else self.gridworld.policy_multiple_actions
        else:
            print("WARNING: multiple actions in the policy found but `multiple_actions` parameter not set appropriately. Changing...")
            multiple_actions = isinstance(policy[0], list)

        for wall_state in grid_positions[CellType.WALL]:
            grid[wall_state.x, wall_state.y] = CellType.WALL
        for goal_state in grid_positions[CellType.GOAL]:
            grid[goal_state.x, goal_state.y] = CellType.GOAL

        fig, ax = plt.subplots(figsize=self.__figsize)

        if show_value_function:
            value_grid = np.zeros_like(grid, dtype=float)
            for idx, pos in enumerate(grid_positions[CellType.NORMAL]):
                value_grid[pos.x, pos.y] = self.gridworld.V[idx]
            
            # Walls should not be affected by the value function color
            for pos in grid_positions[CellType.WALL]:
                value_grid[pos.x, pos.y] = np.nan

            im = ax.imshow(value_grid, cmap="Blues", origin="upper")

            # Colorbar with the same height as the grid
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("Value Function", fontsize=12)

            # Put the walls to its color
            for pos in grid_positions[CellType.WALL]:
                ax.add_patch(plt.Rectangle((pos.y - 0.5, pos.x - 0.5), 1, 1, color=self.WALL_COLOR))
            # Put the goal to its color
            for pos in grid_positions[CellType.GOAL]:
                ax.add_patch(plt.Rectangle((pos.y - 0.5, pos.x - 0.5), 1, 1, color=self.GOAL_COLOR))
            # Put the cliff to its color
            for pos in grid_positions[CellType.CLIFF]:
                ax.add_patch(plt.Rectangle((pos.y - 0.5, pos.x - 0.5), 1, 1, color=self.CLIFF_COLOR))
            
        else:
            # Default color scheme for grid elements
            colormap = {
                CellType.NORMAL: self.NORMAL_COLOR,
                CellType.WALL: self.WALL_COLOR,
                CellType.GOAL: self.GOAL_COLOR
            }
            color_list = [colormap[k] for k in sorted(colormap)]
            cmap = plt.matplotlib.colors.ListedColormap(color_list)
            ax.imshow(grid, cmap=cmap, origin="upper")

        # Add policy arrows
        for idx, pos in enumerate(grid_positions[CellType.NORMAL]):
            actions = policy[idx] if multiple_actions else [policy[idx]]
            y = pos.y
            x = pos.x
            for action in actions:
                dy, dx = self.gridworld.OFFSETS[action]
                ax.quiver(
                    y, x, 0.35 * dy, 0.35 * dx, scale=1, scale_units="xy", angles="xy", 
                    width=0.005, color=self.POLICY_COLOR, headaxislength=3, headlength=3
                )

        # Adjust grid lines and labels
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color="gray", linestyle="-", linewidth=0.5)

        if savefig:
            if save_title is None:
                plt.savefig(os.path.join(self.__out_path, f"{'deterministic' if self.gridworld.deterministic else 'stochastic'}_policy{'_value_function' if show_value_function else ''}"), dpi=300)
            else:
                plt.savefig(os.path.join(self.__out_path, save_title), dpi=300)
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

        

