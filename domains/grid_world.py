# Simple grid, where there may be obstacles or not and the actions are:
# UP (0), RIGHT (1), DOWN (2), LEFT (3)
# (0, 0) is the top left corner

# Postpones the evaluation of the notations. This allows to specify the type of GridWorldMDP
# or GridWorldLMDP in the GridworldEnv class without them having been defined yet.
from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcolors
from matplotlib import colorbar
from models.MDP import MDP
from models.LMDP import LMDP
from models.LMDP_TDR import LMDP_TDR
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from .grid import CustomGrid, CellType, GridWorldActions
from utils.state import State
from typing import Literal
from sys import getsizeof
from scipy.sparse import csr_matrix
from utils.maps import Map
from tqdm import tqdm
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io


class GridWorldEnv:
    def __init__(
        self,
        map: Map,
        max_steps: int | None = None
    ):
        self.custom_grid = CustomGrid("gridworld", map=map)
        self.agent_start_pos = self.custom_grid.start_pos
        
        self.title = map.name
        
        self.max_steps = max_steps
        
        self.__agent_pos = self.agent_start_pos
        
        if max_steps is None:
            self.max_steps = 200

    
    def __get_manual_action(self):
        key_to_action = {
            "up": GridWorldActions.UP, "w": GridWorldActions.UP,
            "down": GridWorldActions.DOWN, "s": GridWorldActions.DOWN,
            "left": GridWorldActions.LEFT, "a": GridWorldActions.LEFT,
            "right": GridWorldActions.RIGHT, "d": GridWorldActions.RIGHT,
            "tab": GridWorldActions.PICKUP,
            "left shift": GridWorldActions.DROP,
            "space": GridWorldActions.TOGGLE
        }

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key).lower()
                    print(key_name)
                    if key_name in key_to_action:
                        return key_to_action[key_name]
    
    
    def state_to_image(self, state: State, plotter: GridWorldPlotter, direction: int) -> pygame.Surface:
        fig, ax = plt.subplots(figsize=plotter.figsize)
        plotter.plot_base_grid(ax, self.custom_grid.positions, state=state, direction=direction)
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        buf.close()
        plt.close(fig)
        return img
    
    
    def play_game(
        self,
        model: GridWorldMDP | GridWorldLMDP,
        policies: list[tuple[int, np.ndarray]],
        manual_play: bool = False,
        num_times: int = 10,
        save_gif: bool = False,
        save_path: str = None
    ):
        if not save_gif:
            pygame.init()
            screen = None
        plotter = GridWorldPlotter(model)
        frames = []
        
        for policy_epoch, policy in policies:
            print(f"Visualizing policy from training epoch: {policy_epoch}")
            for i in tqdm(range(num_times), desc=f"Playing {num_times} games"):
                num_mistakes = 1
                done = False
                actions = 0
                next_properties = {k: v[0] for k, v in self.custom_grid.state_properties.items()}
                next_layout = self.custom_grid.layout_combinations[0]
                self.__agent_pos = self.agent_start_pos
                
                direction = 0 # Agent direction: 0 for vertical, 1 for right and 2 for left

                while True:
                    state = State(self.__agent_pos[0], self.__agent_pos[1], next_layout, **next_properties)
                    state_idx = next(k for k, v in self.custom_grid.state_index_mapper.items() if v == state)
                    
                    frame_img = self.state_to_image(state, plotter, direction)
                    if save_gif:
                        frames.append(frame_img)
                    else:
                        frame_surface = pygame.image.fromstring(
                            frame_img.tobytes(), frame_img.size, frame_img.mode
                        )
                        if screen is None:
                            screen = pygame.display.set_mode(frame_surface.get_size())
                            pygame.display.set_caption("GridWorld Visualization")

                        screen.blit(frame_surface, (0, 0))
                        pygame.display.flip()

                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                return

                    done = self.custom_grid.is_terminal(state)
                    if done: break
                    
                    if manual_play:
                        action = self.__get_manual_action()
                    else:
                        if isinstance(model, GridWorldMDP):
                            if model.deterministic:
                                action = policy[state_idx]
                            else:
                                next_state = np.random.choice(self.custom_grid.get_num_states(), p=model.P[state_idx, policy[state_idx], :])
                                if next_state != np.argmax(model.P[state_idx, policy[state_idx], :]):
                                    print(f"MISTAKE {num_mistakes}")
                                    num_mistakes += 1
                                # We need to get the action that leads to the next state
                                action = model.transition_action(state_idx, next_state)
                        else:
                            next_state = np.random.choice(self.custom_grid.get_num_states(), p=policy[state_idx])
                            print(next_state)
                            if next_state != np.argmax(policy[state_idx]):
                                print(f"MISTAKE {num_mistakes}")
                                num_mistakes += 1
                            action = model.transition_action(state_idx, next_state)
                    
                    next_state, _, _ = self.custom_grid.move(state, action)
                    
                    dx = state.x - next_state.x
                    dy = state.y - next_state.y
                    if dy > 0:
                        # Left
                        direction = 2
                    elif dy < 0:
                        # Right
                        direction = 1
                    else:
                        # Vertical
                        direction = 0
                    
                    if self.custom_grid.is_cliff(state):    
                        break
                    else:
                        self.__agent_pos = (next_state.y, next_state.x)
                    
                    
                    next_properties = next_state.properties
                    next_layout = next_state.layout
                    
                    actions += 1
                    if actions == self.max_steps:
                        break
        
        if save_gif and frames and save_path:
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,
                loop=0
            )
        
        pygame.quit()
                
                    

                
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

    def __init__(
        self,
        map: Map,
        allowed_actions: list[int] = None,
        stochastic_prob: float = 0.9,
        behaviour: Literal["deterministic", "stochastic", "mixed"] = "deterministic",
        mdp: MDP = None
    ):
        """
        Initializes the grid world based on the provided map or generates a simple grid. Also initializes matrices for state transitions (P) and rewards (R).

        Args:
        - grid_size (int, optional): The size of the grid (default is 3). The grid will have a size of grid_size x grid_size.
        - start_pos (tuple[int, int], optional): The starting position of the agent (default is (1, 1)).
        - map (list[str], optional): A custom map represented by a list of strings (default is None). If provided, the grid is loaded from this map.
        - deterministic (bool, optional): Whether the environment is deterministic (default is True).
        - mdp (MDP, optional): A possible instantiation of an MDP object to be used to initialize the superclass of the GridWorldMDP instance.
        """
        
        if mdp is not None:
            assert type(mdp) == MDP, "MDP must be of type mdp"
            
        if allowed_actions:
            self.allowed_actions = allowed_actions
            self.num_actions = len(allowed_actions)
        else:
            self.num_actions = 4
            self.allowed_actions = [i for i in range(self.num_actions)]

        self.stochastic_prob = stochastic_prob
        
        assert behaviour in ["deterministic", "stochastic", "mixed"], f"{behaviour} behaviour not supported."
        self.behaviour = behaviour
        deterministic = self.behaviour == "deterministic"
        
        self.gridworld_env = GridWorldEnv(map=map)
        start_pos = self.gridworld_env.custom_grid.start_pos
        self.start_state = [state for state in self.gridworld_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        # TODO: remove unreachable states in case that I add the possibilities for GridWorld to support objects (e.g. keys, doors, etc.)
        self.remove_unreachable_states()
        
        self.num_states = self.gridworld_env.custom_grid.get_num_states()
        
        
        
        if mdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.gridworld_env.custom_grid.get_num_terminal_states(),
                allowed_actions=self.allowed_actions,
                s0=self.gridworld_env.custom_grid.states.index(self.start_state),
                deterministic=deterministic,
                behaviour=self.behaviour
            )
            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.generate_P(self.gridworld_env.custom_grid, stochastic_prob=self.stochastic_prob)
            
            if map.R is not None:
                assert map.R.shape == self.R.shape, f"Dimensions of custom reward function {map.R.shape} do not match the expected ones: {self.R.shape}"
                self.R = map.R
            else:
                self._generate_R()
            
            print(f"Created MDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
        else:
            # Useful when wanting to create a GridWorldMDP from an embedding of an LMDP into an MDP
            super().__init__(
                num_states=mdp.num_states,
                num_terminal_states=mdp.num_terminal_states,
                allowed_actions=self.allowed_actions,
                s0=mdp.s0,
                deterministic=mdp.deterministic,
                behaviour=self.behaviour
            )
            self.P = mdp.P
            self.R = mdp.R
            

    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the grid world, setting the default reward to -1 for all actions.
        """
        for state in range(self.num_non_terminal_states):
            state_repr = self.gridworld_env.custom_grid.states[state]
            if self.gridworld_env.custom_grid.is_cliff(state_repr):
                self.R[state] = np.full(shape=self.num_actions, fill_value=-50, dtype=np.float64)
            else:
                self.R[state] = np.full(shape=self.num_actions, fill_value=-5, dtype=np.float64)
    
    def remove_unreachable_states(self):
        print("Going to remove unreachable states")
        
        reachable_states = set()
        queue = [self.start_state]

        for terminal_state in queue:
            reachable_states.add(terminal_state)

        while queue:
            current_state = queue.pop(0)
            for action in self.allowed_actions:
                next_state, _, _ = self.gridworld_env.custom_grid.move(current_state, action)
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    queue.append(next_state)

        
        states = [state for state in self.gridworld_env.custom_grid.states if state in reachable_states]
        terminal_states = [state for state in self.gridworld_env.custom_grid.terminal_states if state in reachable_states]

        removed_states = len(self.gridworld_env.custom_grid.states) - len(states)
        print(f"Removing {removed_states} states")

        self.gridworld_env.custom_grid.states = states
        self.gridworld_env.custom_grid.terminal_states = terminal_states
        self.gridworld_env.custom_grid.generate_state_index_mapper()
    
    
    def transition_action(self, state_idx, next_state_idx):
        curr_state = self.gridworld_env.custom_grid.state_index_mapper[state_idx]
        for action in self.allowed_actions:
            move_state, _, _ = self.gridworld_env.custom_grid.move(curr_state, action)
            next_state = self.gridworld_env.custom_grid.state_index_mapper[next_state_idx]
            if type(next_state) == State:
                if move_state == next_state:
                    return action
            else:
                if move_state.y == next_state[0] and move_state.x == next_state[1]:
                    return action
                
        return 0


    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None):
        assert not save_gif or save_path is not None, "Must specify save path"
        if policies is None:
            print("Computing value function...")
            self.compute_value_function()
            self.gridworld_env.play_game(model=self, policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path)
        else:
            self.gridworld_env.play_game(model=self, policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path)
            
    def play_map(self):
        self.gridworld_env.play_game(model=self, policies=[[0, None]], manual_play=True, num_times=100)
    
class GridWorldLMDP(LMDP):
    
    OFFSETS = {
        0: (0, -1),  # UP
        1: (1, 0),   # RIGHT
        2: (0, 1),   # DOWN
        3: (-1, 0)   # LEFT
    }
    
    def __init__(
        self,
        map: Map,
        sparse_optimization: bool = True,
        benchmark_p: bool = False,
        threads: int = 4,
        lmdp: LMDP = None
    ) -> None:
        
        self.deterministic = False
        
        self.allowed_actions = [i for i in range(len(self.OFFSETS))] # It is not that the LMDP has actions, but to determine the transition probabilities, we need to know how the agent moves through the environment
        self.num_actions = len(self.allowed_actions)
        
        
        self.gridworld_env = GridWorldEnv(map=map)
        start_pos = self.gridworld_env.custom_grid.start_pos
        self.start_state = [state for state in self.gridworld_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        # TODO: remove unreachable states in case that I add the possibilities for GridWorld to support objects (e.g. keys, doors, etc.)
        self.num_sates = self.gridworld_env.custom_grid.get_num_states()
        
        if lmdp is None:
            super().__init__(
                self.num_sates,
                num_terminal_states=self.gridworld_env.custom_grid.get_num_terminal_states(),
                s0=self.gridworld_env.custom_grid.states.index(self.start_state),
                sparse_optimization=sparse_optimization
            )
            
            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.p_time = self.generate_P(
                    self.gridworld_env.custom_grid,
                    self.allowed_actions,
                    benchmark=benchmark_p,
                    num_threads=threads
                )
            
            if map.R is not None:
                assert map.R.shape == self.R.shape, f"Dimensions of custom reward function {map.R.shape} do not match the expected ones: {self.R.shape}"
                self.R = map.R
            else:
                self._generate_R()
            
        
        else:
            super().__init__(
                num_states=lmdp.num_states,
                num_terminal_states=lmdp.num_terminal_states,
                s0=lmdp.s0,
                lmbda=lmdp.lmbda,
                sparse_optimization=False #TODO: update
            )
            
            self.P = lmdp.P
            self.R = lmdp.R
    
    
    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the grid world, setting the default reward to -1 for all actions.
        """
        self.R[:] = np.float64(-5)
        cliff_states = [i for i in range(self.num_states) if self.gridworld_env.custom_grid.is_cliff(self.gridworld_env.custom_grid.state_index_mapper[i])]
        self.R[cliff_states] = np.float64(-50)
        self.R[self.num_non_terminal_states:] = np.float64(0)
    
    
    def transition_action(self, state_idx, next_state_idx):
        curr_state = self.gridworld_env.custom_grid.state_index_mapper[state_idx]
        for action in self.allowed_actions:
            move_state, _, _ = self.gridworld_env.custom_grid.move(curr_state, action)
            next_state = self.gridworld_env.custom_grid.state_index_mapper[next_state_idx]
            if type(next_state) == State:
                if move_state == next_state:
                    return action
            else:
                if move_state.y == next_state[0] and move_state.x == next_state[1]:
                    return action
                
        return 0
        
    
    def policy_to_action(self, state: int, next_state: list[int]) -> list[int]:
        origin_x = self.gridworld_env.custom_grid.state_index_mapper[state].x
        origin_y = self.gridworld_env.custom_grid.state_index_mapper[state].y
        actions = []
        for state_idx, state_prob in enumerate(next_state):
            if state_idx < self.num_non_terminal_states:
                tmp_state = self.gridworld_env.custom_grid.states[state_idx]
            else:
                tmp_state = self.gridworld_env.custom_grid.terminal_states[state_idx - self.num_non_terminal_states]
            
            if state_prob == 0: continue
            for action in range(len(self.OFFSETS)):
                dy, dx = self.OFFSETS[action]
                x = tmp_state.x
                y = tmp_state.y
                if (origin_x + dx == x) and (origin_y + dy == y):
                    actions.append(action)
        
        return actions
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None):
        assert not save_gif or save_path is not None, "Must specify save path"
        if policies is None:
            print("Computing value function...")
            self.compute_value_function()
            self.gridworld_env.play_game(model=self, policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path)
        else:
            self.gridworld_env.play_game(model=self, policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path)

            
    def play_map(self):
        self.gridworld_env.play_game(model=self, policies=[[0, None]], manual_play=True, num_times=100)

class GridWorldLMDP_TDR(LMDP_TDR):
    OFFSETS = {
        0: (0, -1),  # UP
        1: (1, 0),   # RIGHT
        2: (0, 1),   # DOWN
        3: (-1, 0)   # LEFT
    }
    
    def __init__(
        self,
        map: Map,
        sparse_optimization: bool = False,  # TODO: check error and correct (does not work with True)
        benchmark_p: bool = False,
        threads: int = 4
    ):
        self.deterministic = False
        self.allowed_actions = [i for i in range(len(self.OFFSETS))] # It is not that the LMDP has actions, but to determine the transition probabilities, we need to know how the agent moves through the environment
        self.num_actions = len(self.allowed_actions)
        
        self.gridworld_env = GridWorldEnv(map=map)
        start_pos = self.gridworld_env.custom_grid.start_pos
        self.start_state = [state for state in self.gridworld_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        # TODO: remove unreachable states in case that I add the possibilities for GridWorld to support objects (e.g. keys, doors, etc.)
        self.num_sates = self.gridworld_env.custom_grid.get_num_states()
        
        super().__init__(
            self.num_sates,
            num_terminal_states=self.gridworld_env.custom_grid.get_num_terminal_states(),
            s0=self.gridworld_env.custom_grid.states.index(self.start_state),
            sparse_optimization=sparse_optimization
        )
        
        if map.P is not None:
            assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
            self.P = map.P
        else:
            self.p_time = self.generate_P(
                self.gridworld_env.custom_grid,
                self.allowed_actions,
                benchmark=benchmark_p,
                num_threads=threads
            )
        
        if map.R is not None:
            assert map.R.shape == self.R.shape, f"Dimensions of custom reward function {map.R.shape} do not match the expected ones: {self.R.shape}"
            self.R = map.R
        else:
            self._generate_R()
        
        print(f"Created LMDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")


    def _generate_R(self):
        if self.sparse_optimization:
            indices = self.P.nonzero()
        else:
            indices = np.where(self.P != 0)
            
        for i, j in zip(indices[0], indices[1]):
            if self.gridworld_env.custom_grid.is_cliff(self.gridworld_env.custom_grid.state_index_mapper[j]) or self.gridworld_env.custom_grid.is_cliff(self.gridworld_env.custom_grid.state_index_mapper[i]):
                self.R[i, j] = np.float64(-50)
            elif self.gridworld_env.custom_grid.is_terminal(self.gridworld_env.custom_grid.state_index_mapper[j]):
                self.R[i, j] = np.float64(0)
            else:
                self.R[i, j] = np.float64(-5)
        
        if self.sparse_optimization:
            print("Converting R into sparse matrix...")
            print(f"Memory usage before conversion: {getsizeof(self.R):,} bytes")
            self.R = csr_matrix(self.R)
            print(f"Memory usage after conversion: {getsizeof(self.R):,} bytes")
                


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
    START_COLOR = "#4EFF10"
    GOAL_COLOR = "#F50735"
    CLIFF_COLOR = "#3F043C"
    POLICY_COLOR = "#FF1010"

    def __init__(self, gridworld: GridWorldMDP | GridWorldLMDP | GridWorldLMDP_TDR, figsize: tuple[int, int] = (5, 5), name: str = "", assets_path: str = None):
        """
        Initializes the GridWorldPlotter with the provided GridWorldMDP instance, figure size, and output directory.

        Args:
        - gridworld (GridWorldMDP): The GridWorldMDP instance that contains the grid world environment and its details.
        - figsize (tuple[int, int], optional): The size of the figure for plotting (default is (5, 5)).
        - name (str, optional): The name for the output directory where the plots will be saved.

        Creates the necessary directories for saving output if they do not exist.
        """
        self.gridworld = gridworld
        self.gridworld_env = gridworld.gridworld_env
        self.is_mdp = isinstance(self.gridworld, MDP)
        self.figsize = figsize
        self.__out_path = os.path.join("assets", name)
        self.__assets_path = assets_path if assets_path else os.path.join("domains", "res")
        if not os.path.exists(self.__out_path): os.makedirs(self.__out_path)
        
        
    def plot_base_grid(self, ax, grid_positions, state: State = None, direction: int = 0, color_start: bool = True, color_goal: bool = True):
        grid = np.full((self.gridworld_env.custom_grid.size_x, self.gridworld_env.custom_grid.size_y), CellType.NORMAL)

        for wall_state in grid_positions[CellType.WALL]:
            grid[wall_state[1], wall_state[0]] = CellType.WALL
        for goal_state in grid_positions[CellType.GOAL]:
            grid[goal_state[1], goal_state[0]] = CellType.GOAL

        # Color grid
        colormap = {
            CellType.NORMAL: self.NORMAL_COLOR,
            CellType.WALL: self.WALL_COLOR,
            CellType.GOAL: self.NORMAL_COLOR
        }
        color_list = [colormap[k] for k in sorted(colormap)]
        cmap = plt.matplotlib.colors.ListedColormap(color_list)
        ax.imshow(grid, cmap=cmap, origin="upper")

        # Start
        if color_start:
            for pos in grid_positions[CellType.START]:
                ax.add_patch(plt.Rectangle((pos[0] - 0.5, pos[1] - 0.5), 1, 1, color=self.START_COLOR))

        # Walls
        for pos in grid_positions[CellType.WALL]:
            ax.add_patch(plt.Rectangle((pos[0] - 0.5, pos[1] - 0.5), 1, 1, color=self.WALL_COLOR))

        # Goal
        if color_goal:
            for pos in grid_positions[CellType.GOAL]:
                ax.add_patch(plt.Rectangle((pos[0] - 0.5, pos[1] - 0.5), 1, 1, color=self.GOAL_COLOR))

        # Cliff
        for pos in grid_positions[CellType.CLIFF]:
            ax.add_patch(plt.Rectangle((pos[0] - 0.5, pos[1] - 0.5), 1, 1, color=self.CLIFF_COLOR))

        
        if state:        
            # Plot the agent
            if direction == 0:
                # Vertical
                agent_filename = "agent_vertical.png"
            elif direction == 1:
                # Right
                agent_filename = "agent_right.png"
            else:
                # Left
                agent_filename = "agent_left.png"
                
            agent_im = mpimg.imread(os.path.join(self.__assets_path, agent_filename))
            agent_x, agent_y = state.x, state.y
            ax.imshow(agent_im, extent=[agent_y - 0.5, agent_y + 0.5, agent_x + 0.5, agent_x - 0.5], zorder=10)
            
            for pos, obj in state.layout.items():
                pos_x = pos[0]
                pos_y = pos[1]
                if obj is None: continue
                if obj.type == "key":
                    img = mpimg.imread(os.path.join(os.path.join(self.__assets_path, "key"), f"{obj.color}_key.png"))
                    ax.imshow(img, extent=[pos_y - 0.5, pos_y + 0.5, pos_x + 0.5, pos_x - 0.5], zorder=10)
                else:
                    # Door
                    img = mpimg.imread(os.path.join(os.path.join(self.__assets_path, "door"), f"{obj.color}_door_{'opened' if state.properties[str(obj)] else 'closed'}.png"))
                    ax.imshow(img, extent=[pos_y - 0.5, pos_y + 0.5, pos_x + 0.5, pos_x - 0.5], zorder=10)
                    
        
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color="black", linestyle="-", linewidth=1)

        
    def __plot_grid_overlays(
        self, 
        ax, 
        divider, 
        show_value_function: bool, 
        policy: np.ndarray, 
        multiple_actions: bool, 
        show_prob: bool, 
        show_actions: bool, 
        prob_size: float, 
        color_probs: bool
    ):
        
        if policy is None:
            if self.is_mdp:
                policy = self.gridworld.policy if not multiple_actions else self.gridworld.policy_multiple_actions
            else:
                policy = self.gridworld.policy if not multiple_actions else self.gridworld.policy_multiple_states
        else:
            if not multiple_actions and isinstance(policy[0], list):
                print("WARNING: multiple actions in the policy found but `multiple_actions` parameter not set appropriately. Changing...")
                multiple_actions = True
                
        grid_positions = self.gridworld_env.custom_grid.positions
        prob_cmap = plt.get_cmap("Greens")

        if show_value_function:
            value_grid = np.zeros((self.gridworld_env.custom_grid.size_x, self.gridworld_env.custom_grid.size_y), dtype=float)
            for idx, pos in enumerate(grid_positions[CellType.NORMAL]):
                value_grid[pos[1], pos[0]] = self.gridworld.V[idx]
            for pos in grid_positions[CellType.WALL]:
                value_grid[pos[1], pos[0]] = np.nan

            im = ax.imshow(value_grid, cmap="Blues", origin="upper")
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("Value Function", fontsize=12)

        for idx, pos in enumerate(grid_positions[CellType.NORMAL]):
            curr_state = [s for s in self.gridworld_env.custom_grid.states if s.x == pos[1] and s.y == pos[0]][0]
            if self.gridworld_env.custom_grid.is_cliff(curr_state): continue
            actions = policy[idx] if multiple_actions else [policy[idx]]

            y, x = curr_state.y, curr_state.x
            if not self.gridworld.deterministic:
                probs = self.gridworld.P[idx, actions[0], :] if self.is_mdp else policy[idx]
                action_probs = self.get_action_probs(curr_state, probs)

                ax.plot([y - 0.5, y + 0.5], [x - 0.5, x + 0.5], color="black", linewidth=0.2)
                ax.plot([y - 0.5, y + 0.5], [x + 0.5, x - 0.5], color="black", linewidth=0.2)

                quadrants = {
                    0: [(y + 0.5, x - 0.5), (y, x), (y - 0.5, x - 0.5)],
                    1: [(y + 0.5, x + 0.5), (y, x), (y + 0.5, x - 0.5)],
                    2: [(y - 0.5, x + 0.5), (y, x), (y + 0.5, x + 0.5)],
                    3: [(y - 0.5, x - 0.5), (y, x), (y - 0.5, x + 0.5)]
                }

                max_prob = max(action_probs.values())
                for action, prob in action_probs.items():
                    q_vertices = quadrants[action]
                    color = prob_cmap(prob)
                    if color_probs:
                        triangle = plt.Polygon(q_vertices, color=color, alpha=0.5)
                        ax.add_patch(triangle)
                    if show_prob:
                        ax.text(np.mean([p[0] for p in q_vertices]), np.mean([p[1] for p in q_vertices]), f"{prob:.2f}",
                                color="white" if self.gridworld.V[idx] > 0.3 * np.min(self.gridworld.V) else "black",
                                fontweight="heavy" if prob == max_prob else None, fontsize=prob_size, ha="center", va="center")
            else:
                if not show_actions: continue
                for action in actions:
                    dy, dx = self.gridworld.OFFSETS[action]
                    ax.quiver(
                        y, x, 0.35 * dy, 0.35 * dx, scale=1, scale_units="xy", angles="xy",
                        width=0.005, color=self.POLICY_COLOR, headaxislength=3, headlength=3
                    )

        if not self.gridworld.deterministic and color_probs:
            cax = divider.append_axes("bottom", size="5%", pad=0.1)
            cbar = colorbar.ColorbarBase(cax, cmap=prob_cmap, orientation='horizontal')
            cbar.set_label("Action Probabilities", fontsize=12)



    def plot_grid_world(
        self,
        savefig: bool = False,
        save_title: str = None,
        show_value_function: bool = False,
        policy: np.ndarray = None,
        multiple_actions: bool = False,
        show_prob: bool = False,
        show_actions: bool = False,
        prob_size: float = None,
        color_probs: bool = True
    ):
        if prob_size is None:
            prob_size = self.figsize[0] / 1


        fig, ax = plt.subplots(figsize=self.figsize)
        divider = make_axes_locatable(ax)
        grid_positions = self.gridworld_env.custom_grid.positions

        self.plot_base_grid(ax, grid_positions)
        if show_value_function or show_prob or show_actions:
            self.__plot_grid_overlays(
                ax, divider,
                show_value_function, policy, multiple_actions,
                show_prob, show_actions, prob_size, color_probs
            )

        if savefig:
            if save_title is None:
                save_title = f"{'deterministic' if self.gridworld.deterministic else 'stochastic'}_policy"
                if show_value_function:
                    save_title += "_value_function"
            plt.savefig(os.path.join(self.__out_path, save_title), dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def plot_state(self, state: State):
        fig, ax = plt.subplots(figsize=self.figsize)
        grid_positions = self.gridworld_env.custom_grid.positions

        self.plot_base_grid(ax, grid_positions, state=state)
        
        plt.show()


    def get_action_probs(self, curr_state: State, probs: list[float]):
        
        x = curr_state.x
        y = curr_state.y
        mapping = {}
        for action in range(self.gridworld.num_actions):
            dy, dx = self.gridworld.OFFSETS[action]
            next_state = State(y + dy, x + dx, curr_state.layout, *curr_state.properties)
            tmp = [k for k, v in self.gridworld_env.custom_grid.state_index_mapper.items() if next_state == v]
            if len(tmp) > 0:
                mapping[action] = probs[tmp[0]]
            # idx_next_state = next(k for k, v in self.gridworld_env.custom_grid.state_index_mapper.items() if next_state == v)
        
        total_prob = sum(mapping.values())
        remaining_actions = self.gridworld.num_actions - len(mapping)
        for action in range(self.gridworld.num_actions):
            if action not in mapping:
                mapping[action] = (1 - total_prob) / remaining_actions
        
        
        return dict(sorted(mapping.items(), key=lambda x: x[0]))
            
            

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

    
    def _get_common_reward_params(self, cmap_name: str = "jet"):
        positions = [val for pos in self.gridworld_env.custom_grid.positions.values() for val in pos]
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=np.min(self.gridworld.R), vmax=np.max(self.gridworld.R))
        
        return positions, cmap, norm
    

    def _get_state_index(self, x, y):
        states = self.gridworld_env.custom_grid.get_state_pos(x, y)
        if not states:
            return None, None
        state = states[0]
        
        return next(k for k, v in self.gridworld_env.custom_grid.state_index_mapper.items() if v == state), state


    def __visualize_reward_mdp_lmdptdr(self, ax, cmap_name: str = "jet"):
        positions, cmap, norm = self._get_common_reward_params(cmap_name=cmap_name)

        for pos in positions:
            y, x = pos
            state_idx, state = self._get_state_index(x, y)
            if state_idx is None or state_idx >= self.gridworld.R.shape[0]:
                continue

            quadrants = {
                0: [(y + 0.5, x - 0.5), (y, x), (y - 0.5, x - 0.5)],
                1: [(y + 0.5, x + 0.5), (y, x), (y + 0.5, x - 0.5)],
                2: [(y - 0.5, x + 0.5), (y, x), (y + 0.5, x + 0.5)],
                3: [(y - 0.5, x - 0.5), (y, x), (y - 0.5, x + 0.5)]
            }


            ax.plot([y - 0.5, y + 0.5], [x - 0.5, x + 0.5], color="black", linewidth=0.2)
            ax.plot([y - 0.5, y + 0.5], [x + 0.5, x - 0.5], color="black", linewidth=0.2)

            for i, verts in quadrants.items():
                if isinstance(self.gridworld, GridWorldMDP):
                    reward = self.gridworld.R[state_idx]
                    reward = reward[i]
                else:
                    if type(self.gridworld.R) == csr_matrix:
                        tmp_R = self.gridworld.R.toarray()
                    else:
                        tmp_R = self.gridworld.R
                    # Will only consider states reached with navigation actions
                    next_state, _, _ = self.gridworld_env.custom_grid.move(state, i)
                    next_state_idx, _ = self._get_state_index(next_state.x, next_state.y)
                    if self.gridworld_env.custom_grid.is_cliff(state):
                        reward = tmp_R[state_idx, np.where(tmp_R[state_idx] != 0)[0]]
                    else:
                        reward = tmp_R[state_idx, next_state_idx]
                    
                triangle = plt.Polygon(verts, color=cmap(norm(reward)), alpha=1)
                ax.add_patch(triangle)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Reward")


    def __visualize_reward_lmdp(self, ax, cmap_name: str = "jet"):
        positions, cmap, norm = self._get_common_reward_params(cmap_name=cmap_name)

        for pos in positions:
            y, x = pos
            state_idx, _ = self._get_state_index(x, y)
            if state_idx is None:
                continue

            reward = self.gridworld.R[state_idx]
            ax.add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, color=cmap(norm(reward))))

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Reward")
    
    
    def visualize_reward(self, savefig: bool = False):
        assert self.gridworld.num_actions == 4, "Gridworld can only have four navigation actions"
        
        plt.rcParams.update({"text.usetex": True})
        fig, ax = plt.subplots(figsize=self.figsize)
        grid_positions = self.gridworld_env.custom_grid.positions

        self.plot_base_grid(ax, grid_positions, color_start=False, color_goal=False)
        
        if isinstance(self.gridworld, GridWorldLMDP):
            self.__visualize_reward_lmdp(ax)
        elif isinstance(self.gridworld, GridWorldMDP) or isinstance(self.gridworld, GridWorldLMDP_TDR):
            self.__visualize_reward_mdp_lmdptdr(ax)
        
        plt.title(self.gridworld_env.title)
        print()
        if savefig:
            plt.savefig(f"assets/reward_{self.gridworld_env.title}_{type(self.gridworld).__name__}.png", dpi=300, bbox_inches="tight")
        else:
            plt.show()