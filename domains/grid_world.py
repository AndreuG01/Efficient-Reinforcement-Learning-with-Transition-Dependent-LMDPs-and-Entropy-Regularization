# Simple grid, where there may be obstacles or not and the actions are:
# UP (0), RIGHT (1), DOWN (2), LEFT (3)
# (0, 0) is the top left corner

# Postpones the evaluation of the notations. This allows to specify the type of GridWorldMDP
# or GridWorldLMDP in the GridworldEnv class without them having been defined yet.
from __future__ import annotations

import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1" # To hide the welcome message of the pygame library
import matplotlib.axes
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcolors
from matplotlib import colorbar
from models.MDP import MDP
from models.LMDP import LMDP
from models.LMDP_TDR import LMDP_TDR
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
from copy import deepcopy
import io
from utils.stats import GameStats


class GridWorldEnv:
    """
    A class that represents the environment for a GridWorld agent to interact with.

    Attributes:
        custom_grid (CustomGrid): The grid representing the world in which the agent navigates.
        agent_start_pos (tuple[int, int]): The starting position of the agent in the grid.
        title (str): The name of the map.
        max_steps (int): The maximum number of steps the agent can take in one episode.
        __agent_pos (tuple[int, int]): The current position of the agent.
    """
    def __init__(
        self,
        map: Map,
        max_steps: int = 200,
        allowed_actions: list[int] = None,
        verbose: bool = True
    ):
        """
        Initializes the GridWorld environment with the provided map and the maximum number of steps.

        Args:
            map (Map): The map representing the environment layout.
            max_steps (int, optional): The maximum number of steps the agent can take. Defaults to 200.
        """
        self.verbose = verbose
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
            
        self.custom_grid = CustomGrid("gridworld", map=map, allowed_actions=self.allowed_actions, verbose=self.verbose)
        self.agent_start_pos = self.custom_grid.start_pos
        
        self.title = map.name
        
        self.max_steps = max_steps
        
        self.__agent_pos = self.agent_start_pos
        
        self.max_steps = 10000#max_steps

    
    def __get_manual_action(self):
        """
        Retrieves a manual action input from the user through keyboard events.

        Returns:
            int: The action corresponding to the user's input.
        """
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
                    self._print(key_name)
                    if key_name in key_to_action:
                        return key_to_action[key_name]
    
    
    def state_to_image(self, state: State, plotter: GridWorldPlotter, direction: int) -> pygame.Surface:
        """
        Converts the current state into an image for visualization.

        Args:
            state (State): The current state of the agent.
            plotter (GridWorldPlotter): The plotter used to visualize the state.
            direction (int): The direction of the agent (0 = vertical, 1 = right, 2 = left).

        Returns:
            pygame.Surface: A surface containing the state visualized as an image.
        """
        fig, axes = plt.subplots(figsize=plotter.figsize)
        plotter.plot_base_grid(axes, self.custom_grid.positions, state=state, direction=direction)
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
        save_path: str = None,
        show_window: bool = True
    ) -> GameStats:
        """
        It allows to visualize the control policy of an agent inf `manual_play` is set to False. Otherwise, the user can
        manually control the agent with the keyboard keys.

        Args:
            model (GridWorldMDP | GridWorldLMDP): The model to use for decision-making (MDP or LMDP).
            policies (list[tuple[int, np.ndarray]]): A list of policies generated at different training epochs.
            manual_play (bool, optional): If True, allows manual control of the agent. Defaults to False.
            num_times (int, optional): The number of times to play the game. Defaults to 10.
            save_gif (bool, optional): If True, saves the game sequence as a GIF. Defaults to False.
            save_path (str, optional): Path to save the GIF if `save_gif` is True.
            show_window (bool, optional): Whether to show the window or not. Defaults to True

        Returns:
            None
        """
        game_stats = deepcopy(GameStats())
        if not save_gif:
            pygame.init()
            screen = None
        plotter = GridWorldPlotter(model)
        frames = []
        # self.max_steps = 30
        
        for policy_epoch, policy in policies:
            self._print(f"Visualizing policy from training epoch: {policy_epoch}")
            for i in tqdm(range(num_times), desc=f"Playing {num_times} games"):
                num_mistakes = 0
                actions = 0
                deaths = 0
                done = False
                next_properties = {k: v[0] for k, v in self.custom_grid.state_properties.items()}
                next_layout = self.custom_grid.layout_combinations[0]
                self.__agent_pos = self.agent_start_pos
                
                direction = 0 # Agent direction: 0 for vertical, 1 for right and 2 for left

                while True:
                    state = State(self.__agent_pos[0], self.__agent_pos[1], next_layout, **next_properties)
                    state_idx = next(k for k, v in self.custom_grid.state_index_mapper.items() if v == state)
                    
                    if show_window:
                        frame_img = self.state_to_image(state, plotter, direction)
                        if save_gif:
                            frames.append(frame_img)
                        elif show_window:
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
                            action = np.random.choice(np.arange(len(policy[state_idx])), p=policy[state_idx].astype(np.float64) if model.dtype == np.float128 else policy[state_idx])
                            next_state = np.random.choice(self.custom_grid.get_num_states(), p=model.P[state_idx, action, :].astype(np.float64) if model.dtype == np.float128 else model.P[state_idx, action, :])
                            if next_state != np.argmax(model.P[state_idx, action, :]):
                                num_mistakes += 1
                                self._print(f"MISTAKE [{num_mistakes} / {actions}]")
                            # We need to get the action that leads to the next state
                            action = self.custom_grid.transition_action(state_idx, next_state, model.allowed_actions)
                        else:
                            # This function uses the C-long dtype, which is 32bit on windows and otherwise 64bit on 64bit platforms (and 32bit on 32bit ones).
                            # Since NumPy 2.0, NumPy’s default integer is 32bit on 32bit platforms and 64bit on 64bit platforms.
                            # Therefore, DO NOT CHANGE THE .astype(np.float64) from the following line.
                            next_state = np.random.choice(self.custom_grid.get_num_states(), p=policy[state_idx].astype(np.float64) if model.dtype == np.float128 else policy[state_idx])
                            if next_state != np.argmax(policy[state_idx]):
                                num_mistakes += 1
                                self._print(f"MISTAKE [{num_mistakes} / {actions}]")
                            action = self.custom_grid.transition_action(state_idx, next_state, model.allowed_actions)
                    
                    next_state, _, _ = self.custom_grid.move(state, action)
                    
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
                        deaths += 1
                        break
                    else:
                        self.__agent_pos = (next_state.y, next_state.x)
                    
                    
                    next_properties = next_state.properties
                    next_layout = next_state.layout
                    
                    actions += 1
                    if actions == self.max_steps:
                        break
            
                game_stats.add_game_info(
                    moves=actions,
                    errors=num_mistakes,
                    deaths=deaths
                )
            
        self._print(game_stats.GAME_INFO)
        
        if save_gif and frames and save_path:
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,
                loop=0
            )
        
        pygame.quit()
        
        return game_stats
    
    def _print(self, msg):
        if self.verbose:
            print(msg)
                
class GridWorldMDP(MDP):
    """
    A class representing a GridWorld Markov Decision Process (MDP).

    This class models a grid world environment where an agent navigates through a grid to reach a goal.
    The grid consists of various types of cells, such as normal cells, walls, a start position, and a goal position.
    The agent's task is to move from the start position to the goal while avoiding walls. The class supports both deterministic and stochastic dynamics for movement.

    Attributes:
        OFFSETS (dict): A dictionary mapping actions to directional offsets.
        num_actions (int): The number of possible actions the agent can take.
        allowed_actions (list[int]): List of allowed actions.
        stochastic_prob (float): Probability of stochastic transitions.
        behaviour (str): The behaviour type for the environment ("deterministic", "stochastic", or "mixed").
        gridworld_env (GridWorldEnv): The environment object representing the grid world.
        start_state (State): The starting state of the agent.
        num_states (int): The total number of states (excluding terminal states).
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
        benchmark_p: bool = False,
        threads: int = 4,
        gamma: float = 1.0,
        temperature: float = 0.0,
        mdp: MDP = None,
        verbose: bool = True,
        dtype: np.dtype = np.float128
    ):
        """
        Initializes the grid world based on the provided map. Based on the allowed actions of the agent, it removes any state from the state space that cannot be reached by the agent.
        If an MDP object is provided, it sets the dynamics to the one from the provided mdp object. Otherwise, it creates default transition probability density and reward function.

        Args:
            map (Map): The map for the environment.
            allowed_actions (list[int], optional): List of allowed actions. Defaults to None.
            stochastic_prob (float, optional): Probability of stochastic transitions. Defaults to 0.9.
            behaviour (Literal["deterministic", "stochastic", "mixed"], optional): Behaviour type for the environment. Defaults to "deterministic".
            threads (int, optional): Number of threads for parallel processing when generating transition probabilities. Defaults to 4.
            gamma (float, optional): The discount factor for the MDP. Defaults to 1.
            mdp (MDP, optional): An existing MDP object to initialize the superclass. Defaults to None.
        """
        self.verbose = verbose
        self.dtype = dtype
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
        
        self.gridworld_env = GridWorldEnv(map=map, allowed_actions=self.allowed_actions, verbose=self.verbose)
        start_pos = self.gridworld_env.custom_grid.start_pos
        self.start_state = [state for state in self.gridworld_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        
        self.num_states = self.gridworld_env.custom_grid.get_num_states()

        
        if mdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.gridworld_env.custom_grid.get_num_terminal_states(),
                allowed_actions=self.allowed_actions,
                s0=self.gridworld_env.custom_grid.states.index(self.start_state),
                deterministic=deterministic,
                behaviour=self.behaviour,
                gamma=gamma,
                temperature=temperature,
                verbose=self.verbose,
                dtype=self.dtype
            )
            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.p_time = self.generate_P(
                    self.gridworld_env.custom_grid,
                    stochastic_prob=self.stochastic_prob,
                    num_threads=threads,
                    benchmark=benchmark_p,
                )
            
            if map.R is not None:
                assert map.R.shape == self.R.shape, f"Dimensions of custom reward function {map.R.shape} do not match the expected ones: {self.R.shape}"
                self.R = map.R
            else:
                self._generate_R()
            
            self._print(f"Created MDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
        else:
            # Useful when wanting to create a GridWorldMDP from an embedding of an LMDP into an MDP
            super().__init__(
                num_states=mdp.num_states,
                num_terminal_states=mdp.num_terminal_states,
                allowed_actions=self.allowed_actions,
                s0=mdp.s0,
                deterministic=mdp.deterministic,
                behaviour=self.behaviour,
                gamma=mdp.gamma,
                temperature=mdp.temperature,
                verbose=mdp.verbose,
                dtype=mdp.dtype
            )
            self.P = mdp.P
            self.R = mdp.R
            

    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the grid world, setting the default reward to -50 for all actions for cliff states and to -5 for normal states. Terminal states get a reward of 0.
        
        Returns:
            None
        """
        for state in range(self.num_non_terminal_states):
            state_repr = self.gridworld_env.custom_grid.states[state]
            if self.gridworld_env.custom_grid.is_cliff(state_repr):
                self.R[state] = np.full(shape=self.num_actions, fill_value=-50, dtype=self.dtype)
            else:
                self.R[state] = np.full(shape=self.num_actions, fill_value=-5, dtype=self.dtype)


    def visualize_policy(
        self,
        policies: list[tuple[int, np.ndarray]] = None,
        num_times: int = 10,
        save_gif: bool = False,
        save_path: str = None,
        show_window: bool = True
    ) -> GameStats:
        """
        Visualizes the policy by running the environment with the given policies. If there are no policies, the optimal one is computed.
        If the `save_gif` parameter is set to true, a GIF with the visualization is stored in the specified path. Otherwise, the policy is displayed to
        the user through the screen.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): List of policies from different epochs. Defaults to None
            num_times (int, optional): Number of times to run the game. Defaults to 10.
            save_gif (bool, optional): If True, saves the game sequence as a GIF. Defaults to False
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.

        Returns:
            None
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if policies is None:
            self._print("Computing value function...")
            self.compute_value_function()
            return self.gridworld_env.play_game(model=self, policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, show_window=show_window)
        else:
            return self.gridworld_env.play_game(model=self, policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, show_window=show_window)
            
    def play_map(self) -> GameStats:
        """
        Allows the user to play with the agent manually through the map.

        Returns:
            None
        """
        return self.gridworld_env.play_game(model=self, policies=[[0, None]], manual_play=True, num_times=100, show_window=True)
    
    
    def to_LMDP_policy(self):
        lmdp_policy = np.zeros((self.num_non_terminal_states, self.num_states), dtype=self.dtype)
        
        for s in range(self.num_non_terminal_states):
            for a in range(self.num_actions):
                next_s, _, terminal = self.gridworld_env.custom_grid.move(self.gridworld_env.custom_grid.states[s], a)
                if not terminal:
                    next_s_idx = self.gridworld_env.custom_grid.states.index(next_s)
                else:
                    next_s_idx = len(self.gridworld_env.custom_grid.states) + self.gridworld_env.custom_grid.terminal_states.index(next_s)
                
                lmdp_policy[s, next_s_idx] += self.policy[s, a]
        
        assert np.all(np.sum(lmdp_policy, axis=1))
        
        return lmdp_policy
    
    
    
    def _print(self, msg):
        if self.verbose:
            print(msg)
    
class GridWorldLMDP(LMDP):
    """
    A class representing a GridWorld linearly-sovable MDP (LMDP).

    This class models a grid world environment where the agent navigates the grid with the objective of maximizing rewards while avoiding cliffs.
    The grid consists of various types of cells, such as normal cells, walls, a start position, and a goal position.
    The agent's task is to move from the start position to the goal while avoiding walls.

    Attributes:
        OFFSETS (dict): A dictionary mapping actions to directional offsets.
        num_actions (int): The number of possible actions the agent can take.
        allowed_actions (list[int]): List of allowed actions.
        gridworld_env (GridWorldEnv): The environment object representing the grid world.
        start_state (State): The starting state of the agent.
        num_sates (int): The total number of states in the grid world (excluding terminal states).
        p_time (float): The time it takes to generate the transition matrix (if applicable).
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
        sparse_optimization: bool = True,
        allowed_actions: list[int] = None,
        benchmark_p: bool = False,
        threads: int = 4,
        lmbda: float = 1.0,
        lmdp: LMDP = None,
        verbose: bool = True,
        dtype: np.dtype = np.float128
    ) -> None:
        """
        Initializes the grid world based on the provided map and optionally inherits from an existing LMDP.

        Args:
            map (Map): The map for the environment.
            allowed_actions (list[int], optional): List of allowed actions. Defaults to None.
            sparse_optimization (bool, optional): Flag indicating whether sparse optimization is enabled. Defaults to True.
            benchmark_p (bool, optional): Flag indicating whether to benchmark the transition probabilities. Defaults to False.
            threads (int, optional): Number of threads for parallel processing when generating transition probabilities. Defaults to 4.
            lmbda (float, optional): The temperature parameter controlling the penalty from the passive dynamics. Defaults to 1.0.
            lmdp (LMDP, optional): An existing LMDP object to initialize the superclass. Defaults to None.
        """
        self.dtype = dtype
        self.deterministic = False
        self.verbose = verbose
        if allowed_actions:
            self.allowed_actions = allowed_actions
            self.num_actions = len(allowed_actions)
        else:
            self.num_actions = 4
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        
        self.gridworld_env = GridWorldEnv(map=map, allowed_actions=self.allowed_actions, verbose=self.verbose)
        start_pos = self.gridworld_env.custom_grid.start_pos
        self.start_state = [state for state in self.gridworld_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        self.num_sates = self.gridworld_env.custom_grid.get_num_states()
        
        if lmdp is None:
            super().__init__(
                self.num_sates,
                num_terminal_states=self.gridworld_env.custom_grid.get_num_terminal_states(),
                s0=self.gridworld_env.custom_grid.states.index(self.start_state),
                sparse_optimization=sparse_optimization,
                lmbda=lmbda,
                verbose=self.verbose,
                dtype=self.dtype
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
                sparse_optimization=False, #TODO: update
                verbose=lmdp.verbose,
                dtype=lmdp.dtype
            )
            
            self.P = lmdp.P
            self.R = lmdp.R
    
    
    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the grid world, setting the default reward to -50 for cliff states and to -5 for normal states.
        Terminal states get a reward of 0.

        Returns:
            None
        """
        self.R[:] = self.dtype(-5)
        cliff_states = [i for i in range(self.num_states) if self.gridworld_env.custom_grid.is_cliff(self.gridworld_env.custom_grid.state_index_mapper[i])]
        self.R[cliff_states] = self.dtype(-50)
        self.R[self.num_non_terminal_states:] = self.dtype(0)
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None) -> GameStats:
        """
        Visualizes the policy by running the environment with the given policies. If there are no policies, the optimal one is computed.
        If the `save_gif` parameter is set to true, a GIF with the visualization is stored in the specified path. Otherwise, the policy is displayed to
        the user through the screen.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): List of policies from different epochs. Defaults to None
            num_times (int, optional): Number of times to run the game. Defaults to 10.
            save_gif (bool, optional): If True, saves the game sequence as a GIF. Defaults to False
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.

        Returns:
            None
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if policies is None:
            self._print("Computing value function...")
            self.compute_value_function()
            return self.gridworld_env.play_game(model=self, policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path)
        else:
            return self.gridworld_env.play_game(model=self, policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path)

            
    def play_map(self) -> None:
        """
        Allows the user to play with the agent manually through the map.

        Returns:
            None
        """
        self.gridworld_env.play_game(model=self, policies=[[0, None]], manual_play=True, num_times=100)

    def _print(self, msg):
        if self.verbose:
            print(msg)
    
class GridWorldLMDP_TDR(LMDP_TDR):
    """
    A class representing a GridWorld LMDP with transition-dependent rewards.

    This class models a grid world environment where the agent navigates through the environment with transition-dependent rewards.
    The grid includes various elements such as cliffs, walls, a start position, and a goal position. The reward function is 
    defined between state transitions and varies depending on the presence of cliffs or goal states.

    Attributes:
        OFFSETS (dict): A dictionary mapping actions to directional offsets.
        num_actions (int): The number of possible actions the agent can take.
        allowed_actions (list[int]): List of allowed actions.
        gridworld_env (GridWorldEnv): The environment object representing the grid world.
        start_state (State): The starting state of the agent.
        num_sates (int): The total number of states in the grid world (excluding terminal states).
        p_time (float): The time it takes to generate the transition matrix (if applicable).
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
        sparse_optimization: bool = False,  # TODO: check error and correct (does not work with True)
        benchmark_p: bool = False,
        lmbda: float = 1.0,
        threads: int = 4,
        lmdp_tdr: LMDP_TDR = None,
        verbose: bool = True,
        dtype: np.dtype = np.float128
    ):
        """
        Initializes the GridWorldTDR environment using the provided map, and sets up the LMDP-TDR structure.

        Args:
            map (Map): The map for the environment.
            allowed_actions (list[int], optional): List of allowed actions. Defaults to None.
            sparse_optimization (bool, optional): Flag indicating whether sparse optimization is enabled. Defaults to False.
            benchmark_p (bool, optional): Flag indicating whether to benchmark the transition probabilities. Defaults to False.
            threads (int, optional): Number of threads for parallel processing when generating transition probabilities. Defaults to 4.
        """
        self.dtype = dtype
        self.deterministic = False
        self.verbose = verbose
        if allowed_actions:
            self.allowed_actions = allowed_actions
            self.num_actions = len(allowed_actions)
        else:
            self.num_actions = 4
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        
        self.gridworld_env = GridWorldEnv(map=map, allowed_actions=self.allowed_actions, verbose=self.verbose)
        start_pos = self.gridworld_env.custom_grid.start_pos
        self.start_state = [state for state in self.gridworld_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        
        self.num_sates = self.gridworld_env.custom_grid.get_num_states()
        
        if lmdp_tdr is None:
        
            super().__init__(
                self.num_sates,
                num_terminal_states=self.gridworld_env.custom_grid.get_num_terminal_states(),
                s0=self.gridworld_env.custom_grid.states.index(self.start_state),
                sparse_optimization=sparse_optimization,
                lmbda=lmbda,
                verbose=self.verbose,
                dtype=self.dtype
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
                num_states=lmdp_tdr.num_states,
                num_terminal_states=lmdp_tdr.num_terminal_states,
                s0=lmdp_tdr.s0,
                lmbda=lmdp_tdr.lmbda,
                sparse_optimization=lmdp_tdr.sparse_optimization,
                verbose=lmdp_tdr.verbose,
                dtype=lmdp_tdr.dtype
            )
            self.P = lmdp_tdr.P
            self.R = lmdp_tdr.R
        
        self._print(f"Created LMDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")


    def _generate_R(self) -> None:
        """
        Generates the transition-based reward matrix (R) for the grid world. 
        Rewards are set as follows:
        - -50 for transitions into or from cliff states.
        - -5 for regular transitions.
        - 0 for transitions into terminal states.

        If sparse optimization is enabled, the reward matrix is converted to a sparse matrix for memory efficiency.

        Returns:
            None
        """
        if self.sparse_optimization:
            indices = self.P.nonzero()
        else:
            indices = np.where(self.P != 0)
            
        for i, j in zip(indices[0], indices[1]):
            if self.gridworld_env.custom_grid.is_cliff(self.gridworld_env.custom_grid.state_index_mapper[j]) or self.gridworld_env.custom_grid.is_cliff(self.gridworld_env.custom_grid.state_index_mapper[i]):
                self.R[i, j] = self.dtype(-50)
            elif self.gridworld_env.custom_grid.is_terminal(self.gridworld_env.custom_grid.state_index_mapper[j]):
                self.R[i, j] = self.dtype(0)
            else:
                self.R[i, j] = self.dtype(-5)
        
        if self.sparse_optimization:
            self._print("Converting R into sparse matrix...")
            self._print(f"Memory usage before conversion: {getsizeof(self.R):,} bytes")
            self.R = csr_matrix(self.R)
            self._print(f"Memory usage after conversion: {getsizeof(self.R):,} bytes")

                
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None, show_window: bool = True) -> GameStats:
        """
        Visualizes the policy by running the environment with the given policies. If there are no policies, the optimal one is computed.
        If the `save_gif` parameter is set to true, a GIF with the visualization is stored in the specified path. Otherwise, the policy is displayed to
        the user through the screen.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): List of policies from different epochs. Defaults to None
            num_times (int, optional): Number of times to run the game. Defaults to 10.
            save_gif (bool, optional): If True, saves the game sequence as a GIF. Defaults to False
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.

        Returns:
            None
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if policies is None:
            self._print("Computing value function...")
            self.compute_value_function()
            return self.gridworld_env.play_game(model=self, policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, show_window=show_window)
        else:
            return self.gridworld_env.play_game(model=self, policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, show_window=show_window)

    def play_map(self) -> None:
        """
        Allows the user to play with the agent manually through the map.

        Returns:
            None
        """
        self.gridworld_env.play_game(model=self, policies=[[0, None]], manual_play=True, num_times=100, show_window=True)


    def _print(self, msg):
        if self.verbose:
            print(msg)
            
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
        Initializes the GridWorldPlotter with a GridWorld environment, figure size, and output directory.

        Args:
            gridworld (GridWorldMDP | GridWorldLMDP | GridWorldLMDP_TDR): An instance representing the GridWorld environment to visualize.
            figsize (tuple[int, int], optional): Size of the matplotlib figure used for plotting. Defaults to (5, 5).
            name (str, optional): Name of the output folder where plots will be saved. Defaults to "".
            assets_path (str, optional): Path to additional assets used for visualization (e.g., icons). Defaults to "domains/res".

        Notes:
            - Automatically creates the output directory if it doesn't exist.
            - Determines if the environment is an MDP-type instance.
        """
        self.gridworld = gridworld
        self.gridworld_env = gridworld.gridworld_env
        self.is_mdp = isinstance(self.gridworld, MDP)
        self.figsize = figsize
        self.__out_path = os.path.join("assets", name)
        self.__assets_path = assets_path if assets_path else os.path.join("domains", "res")
        if not os.path.exists(self.__out_path): os.makedirs(self.__out_path)
        
        
    def plot_base_grid(
        self,
        ax: matplotlib.axes.Axes,
        grid_positions: dict[CellType, list[tuple[int, int]]],
        state: State = None,
        direction: int = 0,
        color_start: bool = True,
        color_goal: bool = True
    ) -> None:
        """
        Plots the base grid layout of the environment, including walls, goals, start positions, cliffs, and optionally the agent.

        This method visualizes the structural elements of the grid (like cell types), and can optionally overlay the agent and
        interactive objects (e.g. keys, doors) based on the current state.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the grid should be drawn.
            grid_positions (dict[CellType, list[tuple[int, int]]]): A dictionary mapping CellType to lists of (x, y) positions for each cell category.
            state (State, optional): The agent's current state, including position and layout of objects. If provided, the agent and its objects are drawn. Defaults to None.
            direction (int, optional): Direction the agent is facing. 0 = vertical. 1 = right. 2 = left. Defaults to 0.
            color_start (bool, optional): Whether to highlight the start position(s). Defaults to True.
            color_goal (bool, optional): Whether to highlight the goal position(s). Defaults to True.
            
        Returns:
            None
        """
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
    ) -> None:
        """
        Overlays the value function, policy, and action probabilities on the grid environment.

        This method visualizes the value function as a heatmap, overlays the policy at each state (either as arrows for deterministic 
        environments or as multiple actions with probabilities for stochastic environments), and optionally displays action probabilities 
        as text or color-coded polygons.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the grid will be drawn.
            divider (mpl_toolkits.axes_grid1.inset_locator.AxesDivider): The divider for adjusting layout and colorbar placement.
            show_value_function (bool): Whether to overlay the value function as a heatmap. Defaults to False.
            policy (np.ndarray): The policy to be visualized. If None, the default policy is used. Defaults to None.
            multiple_actions (bool): Whether to visualize multiple actions per state. Defaults to False.
            show_prob (bool): Whether to display action probabilities at each state. Defaults to False.
            show_actions (bool): Whether to display deterministic actions as arrows. Defaults to True.
            prob_size (float): The font size for displaying action probabilities. Defaults to 10.
            color_probs (bool): Whether to visualize action probabilities with color-coded polygons. Defaults to False.

        Returns:
            None
        """
        
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
            if self.is_mdp:
                if multiple_actions:
                    actions = np.where(policy[idx] != 0)[0]
                else:
                    actions = [np.argmax(policy[idx])]
            else:
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
    ) -> None:
        """
        Plots the grid world environment, optionally displaying the value function, policy, action probabilities, and other elements.

        This method generates a visualization of the grid world, including the structural elements of the environment (e.g., walls, goals, etc.) 
        and overlays the value function, policy, and action probabilities if requested. The resulting plot can either be shown or saved to a file.

        Args:
            savefig (bool, optional): Whether to save the plot as an image file. Defaults to False.
            save_title (str, optional): The filename for saving the plot. If not provided, a default title based on the environment's type will be used. Defaults to None.
            show_value_function (bool, optional): Whether to overlay the value function as a heatmap. Defaults to False.
            policy (np.ndarray, optional): The policy to be visualized. If None, the default policy will be used. Defaults to None.
            multiple_actions (bool, optional): Whether to visualize multiple actions per state. Defaults to False.
            show_prob (bool, optional): Whether to display action probabilities at each state. Defaults to False.
            show_actions (bool, optional): Whether to display deterministic actions as arrows. Defaults to False.
            prob_size (float, optional): The font size for displaying action probabilities. Defaults to None.
            color_probs (bool, optional): Whether to visualize action probabilities with color-coded polygons. Defaults to True.

        Returns:
            None
        """
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
    
    def plot_state(self, state: State) -> None:
        """
        Plots the grid world environment for a specific state.

        This method visualizes the current state of the environment, highlighting the agent's position and the structural elements of the grid.

        Args:
            state (State): The current state of the environment.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        grid_positions = self.gridworld_env.custom_grid.positions

        self.plot_base_grid(ax, grid_positions, state=state)
        
        plt.show()


    def get_action_probs(self, curr_state: State, probs: list[float]) -> dict[int, float]:
        """
        Computes the action probabilities for the given state.

        This method calculates the probabilities of each action from the current state based on the provided probabilities. If the probability for 
        a particular action is missing, it distributes the remaining probability equally among the other actions.

        Args:
            curr_state (State): The current state of the environment, including the agent's position and layout of objects.
            probs (list[float]): A list of probabilities associated with each possible action from the current state.

        Returns:
            dict[int, float]:   A dictionary mapping each action to its respective probability, with actions that were not originally assigned probabilities 
                                receiving an equal share of the remaining probability mass.
        """
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
            savefig (bool, optional): If True, the plot is saved to a file; otherwise, it is displayed. Defaults to False.

        The statistics are based on the `GridWorldMDP` instance's `stats` attribute, which contains relevant data such as cumulative rewards.
        """
        self.gridworld.stats.print_statistics()
        fig = self.gridworld.stats.plot_cum_reward(deterministic=self.gridworld.deterministic, mdp=True)
        if savefig:
            fig.savefig(os.path.join(self.__out_path, f"{'deterministic' if self.gridworld.deterministic else 'stochastic'}_cum_reward.png"))
        else:
            plt.show()

    
    def _get_common_reward_params(self, cmap_name: str = "jet"):
        """
        Retrieves common parameters for visualizing rewards across different types of grid world environments.

        This method gathers the necessary parameters (positions, colormap, and normalization) required for visualizing the reward values 
        in a consistent manner across various grid world configurations.

        Args:
            cmap_name (str, optional): The name of the colormap to be used for visualizing the reward values. Defaults to "jet".

        Returns:
            tuple:
                - list[tuple[int, int]]: A list of (x, y) positions representing the grid cells.
                - matplotlib.colors.Colormap: The colormap object based on the specified cmap_name.
                - matplotlib.colors.Normalize: The normalization object for scaling the reward values.
        """
        positions = [val for pos in self.gridworld_env.custom_grid.positions.values() for val in pos]
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=np.min(self.gridworld.R), vmax=np.max(self.gridworld.R))
        
        return positions, cmap, norm
    
    # TODO: could be moved to the custom grid class
    def _get_state_index(self, x, y) -> tuple[int | None, State | None]:
        """
        Retrieves the state index and the corresponding state at a given grid position.

        This method checks the grid position (x, y) and returns the index of the state within the environment’s state mapping.

        Args:
            x (int): The x-coordinate of the grid cell.
            y (int): The y-coordinate of the grid cell.

        Returns:
            tuple:
                - int or None: The state index, or None if the position does not correspond to a valid state.
                - State or None: The state corresponding to the given position, or None if no state exists at that position.
        """
        states = self.gridworld_env.custom_grid.get_state_pos(x, y)
        if not states:
            return None, None
        state = states[0]
        
        return next(k for k, v in self.gridworld_env.custom_grid.state_index_mapper.items() if v == state), state


    def __visualize_reward_mdp_lmdptdr(self, ax: matplotlib.axes.Axes, cmap_name: str = "jet") -> None:
        """
        Visualizes the reward structure in MDP or LMDP_TDR environments.

        This method visualizes the rewards for each state by drawing polygons with colors based on the reward values, 
        for both MDP and LMDP_TDR environments.

        Args:
            ax (matplotlib.axes.Axes): The axis on which the reward visualization will be drawn.
            cmap_name (str, optional): The name of the colormap used to visualize the reward values. Defaults to "jet".

        Returns:
            None
        """
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
                    # if self.gridworld_env.custom_grid.is_cliff(state):
                    #     reward = tmp_R[state_idx, np.where(tmp_R[state_idx] != 0)[0]]
                    # else:
                    reward = tmp_R[state_idx, next_state_idx]
                    
                triangle = plt.Polygon(verts, color=cmap(norm(reward)), alpha=1)
                ax.add_patch(triangle)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Reward")


    def __visualize_reward_lmdp(self, ax: matplotlib.axes.Axes, cmap_name: str = "jet") -> None:
        """
        Visualizes the reward structure in LMDP environments.

        This method visualizes the rewards for each state in an LMDP environment by drawing colored rectangles, 
        with colors based on the reward values for each state.

        Args:
            ax (matplotlib.axes.Axes): The axis on which the reward visualization will be drawn.
            cmap_name (str, optional): The name of the colormap used to visualize the reward values. Defaults to "jet".

        Returns:
            None
        """
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
    
    
    def visualize_reward(self, ax: matplotlib.axes.Axes = None, savefig: bool = False) -> None:
        """
        Visualizes the reward structure of the grid world environment.

        This method generates a plot showing the rewards for each state in the environment. The reward values are visualized 
        using color-coded polygons or rectangles, depending on the grid world configuration (MDP, LMDP, or LMDP_TDR). The plot 
        can either be displayed or saved as an image file.

        Args:
            savefig (bool, optional): Whether to save the generated plot as an image file. Defaults to False.
            axes (matplotlib.axses.Axes, optional): An axis where the reward should be plotted. Defaults to None.

        Returns:
            None
        """
        assert self.gridworld.num_actions == 4, "Gridworld can only have four navigation actions"
        create_ax = ax == None
        if create_ax:
            plt.rcParams.update({"text.usetex": True})
            fig, ax = plt.subplots(figsize=self.figsize)
        grid_positions = self.gridworld_env.custom_grid.positions

        self.plot_base_grid(ax, grid_positions, color_start=False, color_goal=False)
        
        if isinstance(self.gridworld, GridWorldLMDP):
            self.__visualize_reward_lmdp(ax)
        elif isinstance(self.gridworld, GridWorldMDP) or isinstance(self.gridworld, GridWorldLMDP_TDR):
            self.__visualize_reward_mdp_lmdptdr(ax)
        
        if create_ax:
            plt.title(self.gridworld_env.title)
        
            if savefig:
                plt.savefig(f"assets/reward_{self.gridworld_env.title}_{type(self.gridworld).__name__}.png", dpi=300, bbox_inches="tight")
            else:
                plt.show()