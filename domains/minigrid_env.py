from __future__ import annotations

from copy import deepcopy
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from .grid import CustomGrid, CellType, MinigridActions
from models.MDP import MDP
from models.LMDP import LMDP
from models.LMDP_TDR import LMDP_TDR
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.state import State, Object
from utils.maps import Map
from tqdm import tqdm
from collections.abc import Callable
from scipy.sparse import csr_matrix
from sys import getsizeof
from typing import Literal
from utils.stats import GameStats



class CustomMinigridEnv(MiniGridEnv):
    """
    A custom environment that extends MiniGridEnv to incorporate a more flexible grid system with different cell types,
    including walls, goals, and cliffs, along with agent orientation and other properties.

    Directions:
        0 --> RIGHT
        1 --> DOWN
        2 --> LEFT
        3 --> UP

    Attributes:
        custom_grid (CustomGrid): A grid representation with different cell types and properties.
        agent_start_pos (tuple[int, int]): The starting position of the agent in the grid.
        agent_start_dir (int): The starting direction (orientation) of the agent. Can be one of [0, 1, 2, 3] representing right, down, left, and up respectively.
        title (str): The name of the grid, used for visualization purposes.
        max_steps (int): The maximum number of steps allowed for the agent to take during an episode.
    """
    def __init__(
        self,
        map: Map,
        properties: dict[str, list] = None,
        agent_start_dir=0,
        max_steps: int = 100000, # Update depending on the needs.
        allowed_actions: list[int] = None,
        verbose: bool = True,
        **kwargs,
    ):
        self.verbose = verbose
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        
        self.custom_grid = CustomGrid("minigrid", map=map, properties=properties, allowed_actions=allowed_actions, verbose=self.verbose)
        self.agent_start_pos = self.custom_grid.start_pos
        self.agent_start_dir = agent_start_dir
        
        self.title = map.name

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.max_steps = max_steps

        super().__init__(
            mission_space=mission_space,
            # grid_size=size,
            highlight=False, # To avoid seeing the agent's view, which is not considered in our models.
            see_through_walls=False,
            max_steps=self.max_steps,
            width=self.custom_grid.size_y,
            height=self.custom_grid.size_x,
            **kwargs,
        )
        

    @staticmethod
    def _gen_mission():
        return "grand mission"

    
    def _gen_grid(self, width, height):
        """
        Overwrites the superclass' method to generate the custom MiniGrid's grid.
        It initializes the interal structures used bythe gym library to visualize and manage the grid environment.
        
        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
        """
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate the walls
        for pos in self.custom_grid.positions[CellType.WALL]:
            self.grid.set(pos[0], pos[1], Wall())
        
        first_goal = self.custom_grid.goal_pos[0]
        self.put_obj(Goal(), first_goal[0], first_goal[1])
        for goal_state in self.custom_grid.goal_pos:
            if goal_state[0] != first_goal[0] and goal_state[1] != first_goal[1]:
                self.put_obj(Goal(), goal_state[0], goal_state[1])
        
        # TODO: modify to acount for the correct key-door mapping
        for object in self.custom_grid.objects:
            if object.type == "door":
                self.grid.set(object.x, object.y, Door(object.color, is_locked=True))
            elif object.type == "key":
                self.grid.set(object.x, object.y, Key(object.color))        
        
        
        for pos in self.custom_grid.positions[CellType.CLIFF]:
            self.put_obj(Lava(), pos[0], pos[1])
        
        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = self.title
    
    
    def _add_frame_with_title(self, frame: np.ndarray, title_text: str) -> Image:
        """
        Adds a title to the given frame and returns the modified frame.

        Args:
            frame (np.ndarray): The frame to which the title will be added.
            title_text (str): The text to display as the title.

        Returns:
            Image: The modified frame with the title.
        """
        curr_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(curr_frame)
        
        title_height = 40
        frame_with_title = Image.new("RGB", (curr_frame.width, curr_frame.height + title_height), "white")
        frame_with_title.paste(curr_frame, (0, title_height))
        
        # Add the text to the frame
        draw_title = ImageDraw.Draw(frame_with_title)
        font = ImageFont.load_default()
        # font = ImageFont.truetype("UbuntuMono.ttf", size=20)
        text_bbox = draw_title.textbbox((0, 0), title_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = ((frame_with_title.width - text_width) // 2, (title_height - text_height) // 2)
        draw_title.text(text_position, title_text, fill="black", font=font)
        
        return frame_with_title

    def visualize_policy(
        self,
        model: MinigridLMDP | MinigridMDP,
        policies: list[tuple[int, np.ndarray]],
        num_times: int=10,
        save_gif: bool = False,
        save_path: str = None,
        show_window: bool = True,
        title: str = None,
    ) -> GameStats:
        """
        Visualizes the behavior of the agent under some given policies by running multiple episodes, rendering each step, 
        and optionally saving the resulting frames as a GIF.

        Args:
            model (MinigridLMDP | MinigridMDP): The agent's model. It is used to determine the sequence of states that need to be visualized when following a policy.
            policies (list[tuple[int, np.ndarray]]): A list of policy arrays, one for each possible policy to visualize. Each policy contains the training epoch from which it was derived.
            num_times (int): The number of times to run each policy (default is 10).
            save_gif (bool): Whether to save the visualization as a GIF (default is False).
            save_path (str): The path to save the GIF if `save_gif` is True.
            show_window (bool, optional): Whether to show the window when playing the game or not. Defaults to False.
            title (str, optional): The title to display on the rendered frames. If None, the default title from the map will be used.
        
        Returns:
            GameStats: An object containing statistics about the game played, such as number of moves, errors, and deaths.
        """
        game_stats = deepcopy(GameStats())
        frames = []
        if not save_gif:
            self.render_mode = "human"
        if not show_window:
            self.render_mode = "rgb_array"
        
        for policy_epoch, policy in policies:
            self._print(f"Visualizing policy from training epoch: {policy_epoch}")
            for i in range(num_times):
                game_title = f"Epoch: {policy_epoch}" if title is None else title
                num_mistakes = 0
                actions = 0
                deaths = 0
                self.reset()
                done = False
                next_properties = {k: v[0] for k, v in self.custom_grid.state_properties.items()}
                
                next_layout = self.custom_grid.layout_combinations[0]
                while not done:
                    state = State(self.agent_pos[1], self.agent_pos[0], next_layout, **next_properties)
                    state_idx = next(k for k, v in self.custom_grid.state_index_mapper.items() if v == state)
                    
                    if isinstance(model, MinigridMDP):
                        action = np.random.choice(np.arange(len(policy[state_idx])), p=policy[state_idx].astype(np.float64) if model.dtype == np.float128 else policy[state_idx])
                        # action = np.argmax(policy[state_idx])
                        next_state = np.random.choice(self.custom_grid.get_num_states(), p=model.P[state_idx, action, :].astype(np.float64) if model.dtype == np.float128 else model.P[state_idx, action, :])
                        
                        if next_state != np.argmax(model.P_det[state_idx, action, :]):
                            num_mistakes += 1
                            self._print(f"Game {i}. [{num_mistakes} mistakes / {actions} total actions]".ljust(50), end="\r")
                        
                        # We need to get the action that leads to the next state
                        action = self.custom_grid.transition_action(state_idx, next_state, model.allowed_actions)
                            
                    else:
                        # next_state = np.argmax(policy[state_idx])
                        next_state = np.random.choice(self.custom_grid.get_num_states(), p=policy[state_idx].astype(np.float64) if model.dtype == np.float128 else policy[state_idx])
                        if next_state != np.argmax(policy[state_idx]):
                            num_mistakes += 1
                            self._print(f"Game {i}. [{num_mistakes} mistakes / {actions} total actions]".ljust(50), end="\r")
                        action = self.custom_grid.transition_action(state_idx, next_state, model.allowed_actions)
                    
                    next_state, _, terminal = self.custom_grid.move(state, action)
                    
                    if self.custom_grid.is_cliff(state):
                        deaths += 1
                    
                    next_properties = next_state.properties
                    next_layout = next_state.layout
                    
    
                    if save_gif:
                        frame = self.render()
                        frames.append(Image.fromarray(frame))
                    _, _, done, _, info = self.step(action)
                    
                    actions += 1
                    if actions == self.max_steps:
                        break
            
                game_stats.add_game_info(
                    moves=actions,
                    errors=num_mistakes,
                    deaths=deaths
                )
                if save_gif:
                    # Add the last frame with title
                    frame = self.render()
                    frames.append(Image.fromarray(frame))
        
        self._print(game_stats.GAME_INFO)
        if not save_gif:
            self.close()
        
        if save_gif:
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=200,
                loop=0  # Loop forever
            )
    
        return game_stats
    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)
    
    
    def visualize_state(self, state: State, save_path: str = None, title: str = None) -> Image:
        """
        Visualizes a given state by rendering the environment at that state.

        Args:
            state (State): The state to visualize.
            save_path (str): Optional path to save the rendered image. If None, the image will not be saved.
            title (str): Optional title to overlay on the rendered image.

        Returns:
            Image: The rendered image of the environment at the specified state.
        """
        self.grid = Grid(self.custom_grid.size_y, self.custom_grid.size_x)
        self.grid.wall_rect(0, 0, self.custom_grid.size_y, self.custom_grid.size_x)
        
        # Generate the walls
        for pos in self.custom_grid.positions[CellType.WALL]:
            self.grid.set(pos[0], pos[1], Wall())
        
        first_goal = self.custom_grid.goal_pos[0]
        self.put_obj(Goal(), first_goal[0], first_goal[1])
        for goal_state in self.custom_grid.goal_pos:
            if goal_state[0] != first_goal[0] and goal_state[1] != first_goal[1]:
                self.put_obj(Goal(), goal_state[0], goal_state[1])
        
        for pos, object in state.layout.items():
            if object is not None:
                if object.type == "door":
                    self.grid.set(pos[1], pos[0], Door(object.color, is_open=state.properties[str(object)], is_locked=True))
                elif object.type == "key":
                    self.grid.set(pos[1], pos[0], Key(object.color))
        
        for pos in self.custom_grid.positions[CellType.CLIFF]:
            self.put_obj(Lava(), pos[0], pos[1])
        
        # Place the agent
        self.agent_pos = (state.x, state.y)
        self.agent_dir = state.properties["orientation"]
        
        
        frame = self.render()
        
        if title is not None:
            img = self._add_frame_with_title(frame, title)
        else:
            img = Image.fromarray(frame)
        
        if save_path:
            img.save(save_path)
            
        return img


class MinigridMDP(MDP):
    """
    A specialized Markov Decision Process (MDP) for MiniGrid environments.

    This class constructs an MDP representation from a MiniGrid-based map. It supports deterministic, stochastic,
    and mixed behavior (stochastic for navigation and deterministic for manipulation) modes.

    Attributes:
        OFFSETS (dict[int, tuple[int, int]]): Maps direction indices to (dx, dy) movement.
        stochastic_prob (float): Probability of following the intended action in a stochastic setting.
        behavior (Literal["deterministic", "stochastic", "mixed"]): Transition behavior type. Defaults to "deterministic".
        num_actions (int): Number of allowed actions.
        allowed_actions (list[int]): List of action indices.
        environment (CustomMinigridEnv): MiniGrid environment wrapper.
        start_state (State): Initial state of the agent.
        num_states (int): The total numbe rof states (excluding terminal states).
        p_time (float): Time taken to generate the transition probability matrix.
        P_det (np.ndarray): Deterministic transition probability matrix.
    """
    
    OFFSETS = {
        0: (1, 0),   # RIGHT
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (0, -1)   # UP
    }
    
    def __init__(
        self,
        map: Map,
        allowed_actions: list[int] = None,
        properties: dict[str, list] = {"orientation": [i for i in range(4)]},
        stochastic_prob: float = 0.9,
        behavior: Literal["deterministic", "stochastic", "mixed"] = "deterministic",
        benchmark_p: bool = False,
        threads: int = 4,
        gamma: float = 1.0,
        temperature: float = 0.0,
        mdp: MDP = None,
        verbose: bool = True,
        dtype: np.dtype = np.float128
    ):
        """
        Initializes a MinigridMDP instance.

        Args:
            map (Map): The map defining the MiniGrid layout.
            allowed_actions (list[int], optional): List of allowed action indices. If None, defaults to [0, 1, 2].
            properties (dict[str, list], optional): State properties for initialization. Defaults to {"orientation": [0, 1, 2, 3]}.
            stochastic_prob (float): Probability of following the intended action in a stochastic setting. Defaults to 0.9.
            behavior (str): One of "deterministic", "stochastic", or "mixed". Defaults to "deterministic".
            gamma (float, optional): The discount factor for the MDP. Defaults to 1.0
            mdp (MDP, optional): If provided, initializes this MinigridMDP using an existing MDP's parameters. Defaults to None.
        """
        self.dtype = dtype
        self.verbose = verbose
        self.stochastic_prob = stochastic_prob
        assert behavior in ["deterministic", "stochastic", "mixed"], f"{behavior} behavior not supported."
        self.behavior = behavior
        deterministic = self.behavior == "deterministic"
        
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.environment = CustomMinigridEnv(render_mode="rgb_array", map=map, properties=properties, allowed_actions=self.allowed_actions, verbose=self.verbose)
        start_pos = self.environment.custom_grid.start_pos
        self.start_state = [state for state in self.environment.custom_grid.states if state.x == start_pos[0] and state.y == start_pos[1]][0]
        
        
        self.num_states = self.environment.custom_grid.get_num_states()
        
        self._print(f"MDP with {self.num_actions} actions. Allowed actions: {self.allowed_actions}")
        

        if mdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.environment.custom_grid.get_num_terminal_states(),
                allowed_actions=self.allowed_actions,
                s0=self.environment.custom_grid.states.index(self.start_state),
                deterministic=deterministic,
                behavior=self.behavior,
                gamma=gamma,
                temperature=temperature,
                verbose=self.verbose,
                dtype=self.dtype
            )
            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.P, self.p_time = self.generate_P(
                    self.environment.custom_grid,
                    stochastic_prob=self.stochastic_prob,
                    benchmark=benchmark_p,
                    num_threads=threads
                )
                
            self.P_det, _ = self.generate_P(
                self.environment.custom_grid,
                stochastic_prob=1,
                benchmark=False,
                num_threads=threads
            )
            
            # If the agent has a mixed behavior, we have to make navigation actions deterministic.
            if self.behavior == "mixed":
                manip_start = MinigridActions.PICKUP
                states = np.arange(self.num_non_terminal_states)
                manip_probs = self.P[states, manip_start:, :]
                max_indices = np.argmax(manip_probs, axis=2)

                self.P[states[:, None], manip_start + np.arange(max_indices.shape[1]), :] = 0
                self.P[states[:, None], manip_start + np.arange(max_indices.shape[1]), max_indices] = 1

            if map.R is not None:
                assert map.R.shape == self.R.shape, f"Dimensions of custom reward function {map.R.shape} do not match the expected ones: {self.R.shape}"
                self.R = map.R
            else:
                self._generate_R()
            
            self._print(f"Created MDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
        else:
            # Useful when wanting to create a MinigridMDP from an embedding of an LMDP into an MDP
            super().__init__(
                num_states=mdp.num_states,
                num_terminal_states=mdp.num_terminal_states,
                allowed_actions=self.allowed_actions,
                s0=mdp.s0,
                gamma=mdp.gamma,
                deterministic=mdp.deterministic,
                behavior=self.behavior,
                temperature=mdp.temperature,
                verbose=mdp.verbose,
                dtype=mdp.dtype
            )
            
            self.P = mdp.P
            self.R = mdp.R
        
          
    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the minigrid, setting the default reward to -50 for all actions for cliff states and to -5 for normal states.
        Terminal states get a reward of 0.

        Returns:
            None
        """
        for state in range(self.num_non_terminal_states):
            state_repr = self.environment.custom_grid.states[state]
            if self.environment.custom_grid.is_cliff(state_repr):
                # For precision purposes, do not use rewards non strictily lower than np.log(np.finfo(np.float128).tiny) = -708
                self.R[state] = np.full(shape=self.num_actions, fill_value=-50, dtype=self.dtype)
            else:
                self.R[state] = np.full(shape=self.num_actions, fill_value=-5, dtype=self.dtype)

    
    def states_to_goal(self, include_actions: bool = False) -> list[int] | tuple[list[int], list[int]]:
        """
        Returns the sequence of state indices that lead to the goal according to the current policy.

        Args:
            include_actions (bool): If True, also returns the actions taken to make the transitions.

        Returns:
            list[int] or tuple: A list of state indices, or a tuple (states, actions).
        """
        curr_state = self.environment.custom_grid.state_index_mapper[self.s0]
        curr_state_idx = self.s0
        states = [self.s0]
        actions = []
        
        
        while not self.environment.custom_grid.is_terminal(curr_state):
            if self.behavior == "stochastic":
                curr_state_idx = np.random.choice(self.environment.custom_grid.get_num_states(), p=self.P[curr_state_idx, self.policy[curr_state_idx], :].astype(np.float64) if self.dtype == np.float128 else self.P[curr_state_idx, self.policy[curr_state_idx], :])
                curr_state = self.environment.custom_grid.state_index_mapper[curr_state_idx]
            elif self.behavior == "deterministic":
                curr_action = self.policy[curr_state_idx][0]
                print(curr_state)
                curr_state, _, terminal = self.environment.custom_grid.move(self.environment.custom_grid.state_index_mapper[curr_state_idx], curr_action)
                if terminal:
                    curr_state_idx = self.environment.custom_grid.terminal_states.index(curr_state)
                else:
                    curr_state_idx = self.environment.custom_grid.states.index(curr_state)

                
                actions.append(curr_action)
            states.append(curr_state_idx)
        
        if include_actions:
            return (states, actions)
        
        return states


    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None, show_window: bool = True) -> GameStats:
        """
        Visualizes the learnt policy on the MiniGrid environment.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): Custom policies to visualize. Each tuple is (step, policy). Defaults to None.
            num_times (int): Number of times to run the game. Defaults to 10.
            save_gif (bool): If True, saves the game sequence as a GIF. Defaults to False.
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.
            show_window (bool): If True, shows the game window while playing. Defaults to True.
        
        Returns:
            GameStats: An object containing statistics about the game played, such as number of moves, errors, and deaths.
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if policies is None:
            self._print(f"Computing value function...")
            self.compute_value_function()
            return self.environment.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self, show_window=show_window, title="Value Iteration policy")
        else:
            return self.environment.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self, show_window=show_window)

    
    def to_LMDP_policy(self) -> np.ndarray:
        """
        Converts the MDP policy to an LMDP policy.
        
        Returns:
            np.ndarray: The LMDP policy.
        """
        lmdp_policy = np.einsum("sa,sap->sp", self.policy[:self.num_non_terminal_states, :], self.P)
        
        assert np.all(np.sum(lmdp_policy, axis=1))
        
        return lmdp_policy
    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)

    def __eq__(self, obj):
        raise NotImplementedError("Method not implemented")

class MinigridLMDP(LMDP):
    """
    A specialized linearly-solvable MDP for MiniGrid environments.

    This class constructs an LMDP representation from a MiniGrid-based map.
    
    Attributes:
        OFFSETS (dict[int, tuple[int, int]]): Maps direction indices to (dx, dy) movement.
        num_actions (int): Number of allowed actions.
        environment (CustomMinigridEnv): MiniGrid environment wrapper.
        allowed_actions (list[int]): List of action indices.
        start_state (State): Initial state of the agent.
        num_states (int): The total number of states (excluding terminal states).
        p_time (float): Time taken to generate the transition probability matrix.
    """
    
    OFFSETS = {
        0: (1, 0),   # RIGHT
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (0, -1)   # UP
    }
    
    def __init__(
        self,
        map: Map,
        allowed_actions: list[int] = None,
        properties: dict[str, list] = {"orientation": [i for i in range(4)]},
        sparse_optimization: bool = True,
        benchmark_p: bool = False,
        threads: int = 4,
        lmbda: float = 1.0,
        lmdp: LMDP = None,
        verbose: bool = True,
        dtype: np.dtype = np.float128
    ):
        """
        Initializes a MinigridLMDP instance.

        Args:
            map (Map): The map defining the MiniGrid layout.
            allowed_actions (list[int], optional): List of allowed action indices. If None, defaults to [0, 1, 2].
            properties (dict[str, list], optional): State properties for initialization. Defaults to {"orientation": [0, 1, 2, 3]}.
            sparse_optimization (bool): Whether to use sparse matrix optimization. Defaults to False.
            benchmark_p (bool): Whether to time the transition probability matrix generation. Defaults to False.
            threads (int): Number of threads for the transition probability matrix computation. Defaults to 4.
            lmdp (LMDP, optional): If provided, initializes this MinigridLMDP using an existing LMDP's parameters. Defaults to None.
        """
        self.dtype = dtype
        self.verbose = verbose
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.environment = CustomMinigridEnv(render_mode="rgb_array", map=map, properties=properties, allowed_actions=self.allowed_actions, verbose=self.verbose)
        
        start_pos = self.environment.custom_grid.start_pos
        self.start_state = [state for state in self.environment.custom_grid.states if state.x == start_pos[0] and state.y == start_pos[1]][0]
        
        self.num_states = self.environment.custom_grid.get_num_states()
        
        
        if lmdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.environment.custom_grid.get_num_terminal_states(),
                s0=self.environment.custom_grid.states.index(self.start_state),
                lmbda=lmbda,
                sparse_optimization=sparse_optimization,
                verbose=self.verbose,
                dtype=self.dtype
            )

            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.p_time = self.generate_P(
                    self.environment.custom_grid,
                    self.allowed_actions,
                    benchmark=benchmark_p,
                    num_threads=threads
                )
            
            if map.R is not None:
                assert map.R.shape == self.R.shape, f"Dimensions of custom reward function {map.R.shape} do not match the expected ones: {self.R.shape}"
                self.R = map.R
            else:
                self._generate_R()
            
            self._print(f"Created LMDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
                    
        else:
            super().__init__(
                num_states=lmdp.num_states,
                num_terminal_states=lmdp.num_terminal_states,
                s0=lmdp.s0,
                lmbda=lmdp.lmbda,
                sparse_optimization=lmdp.sparse_optimization,
                verbose=lmdp.verbose,
                dtype=lmdp.dtype
            )
            self.P = lmdp.P
            self.R = lmdp.R

                  
    def _generate_R(self) -> None:
        """
        Generates the reward matrix (R) for the minigrid, setting the default reward to -50 for cliff states and to -5 for normal states.
        Terminal states get a reward of 0.

        Returns:
            None
        """
        self.R[:] = self.dtype(-5)
        cliff_states = [i for i in range(self.num_states) if self.environment.custom_grid.is_cliff(self.environment.custom_grid.state_index_mapper[i])]
        self.R[cliff_states] = self.dtype(-50)
        self.R[self.num_non_terminal_states:] = self.dtype(0)

    
    def states_to_goal(self, stochastic: bool = False) -> list[int]:
        """
        Computes the trajectory of state indices from start to goal based on the current policy.

        Args:
            stochastic (bool): Whether to sample actions probabilistically or use greedy strategy. Defaults to False.

        Returns:
            list[int]: Sequence of state indices leading to the goal.
        """
        curr_state = self.environment.custom_grid.state_index_mapper[self.s0]
        curr_state_idx = self.s0
        states = [self.s0]
        
        while not self.environment.custom_grid.is_terminal(curr_state):        
            if stochastic:
                curr_state_idx = np.random.choice(self.environment.custom_grid.get_num_states(), p=self.policy[curr_state_idx].astype(np.float64) if self.dtype == np.float128 else self.policy[curr_state_idx])
            else:
                curr_state_idx = np.argmax(self.policy[curr_state_idx])
            
            print(curr_state)
            curr_state = self.environment.custom_grid.state_index_mapper[curr_state_idx]
            states.append(curr_state_idx)
        
        return states
        
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None, show_window: bool = True) -> GameStats:
        """
        Visualizes the learnt policy on the MiniGrid environment.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): Custom policies to visualize. Each tuple is (step, policy). Defaults to None.
            num_times (int): Number of times to run the game. Defaults to 10.
            save_gif (bool): If True, saves the game sequence as a GIF. Defaults to False.
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.
            show_window (bool): If True, shows the game window while playing. Defaults to True.
        
        Returns:
            GameStats: An object containing statistics about the game played, such as number of moves, errors, and deaths.
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            self._print(f"Computing value function...")
            self.compute_value_function()
            return self.environment.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self, show_window=show_window, title="Power Iteration policy")
        else:
            assert policies is not None
            return self.environment.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self, show_window=show_window)
    
    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)
    
    
    def __eq__(self, obj):
        raise NotImplementedError("Method not implemented")

class MinigridLMDP_TDR(LMDP_TDR):
    """
    A specialized LMDP with transition dependent-rewards for MiniGrid environments.

    This class constructs an LMDP representation from a MiniGrid-based map.
    
    Attributes:
        OFFSETS (dict[int, tuple[int, int]]): Maps direction indices to (dx, dy) movement.
        num_actions (int): Number of allowed actions.
        environment (CustomMinigridEnv): MiniGrid environment wrapper.
        allowed_actions (list[int]): List of action indices.
        start_state (State): Initial state of the agent.
        num_states (int): The total number of states (excluding terminal states).
    """
    
    OFFSETS = {
        0: (1, 0),   # RIGHT
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (0, -1)   # UP
    }
    
    def __init__(
        self,
        map: Map,
        allowed_actions: list[int] = None,
        properties: dict[str, list] = {"orientation": [i for i in range(4)]},
        sparse_optimization: bool = True,
        benchmark_p: bool = False,
        threads: int = 4,
        verbose: bool = True,
        lmdp: LMDP_TDR = None,
        dtype: np.dtype = np.float128
    ):
        """
        Initializes a MinigridLMDP_TDR instance.

        Args:
            map (Map): The map defining the MiniGrid layout.
            allowed_actions (list[int], optional): List of allowed action indices. If None, defaults to [0, 1, 2].
            properties (dict[str, list], optional): State properties for initialization. Defaults to {"orientation": [0, 1, 2, 3]}.
            sparse_optimization (bool): Whether to use sparse matrix optimization. Defaults to False.
            benchmark_p (bool): Whether to time the transition probability matrix generation. Defaults to False.
            threads (int): Number of threads for the transition probability matrix computation. Defaults to 4.
        """
        self.dtype = dtype
        self.verbose = verbose
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.environment = CustomMinigridEnv(render_mode="rgb_array", map=map, properties=properties, allowed_actions=self.allowed_actions, verbose=self.verbose)
        start_pos = self.environment.custom_grid.start_pos
        self.start_state = [state for state in self.environment.custom_grid.states if state.x == start_pos[0] and state.y == start_pos[1]][0]
        
        self.num_states = self.environment.custom_grid.get_num_states()
        
        if lmdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.environment.custom_grid.get_num_terminal_states(),
                s0=self.environment.custom_grid.states.index(self.start_state),
                sparse_optimization=sparse_optimization,
                verbose=self.verbose,
                dtype=self.dtype
            )

            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.p_time = self.generate_P(
                    self.environment.custom_grid,
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
                lmbda=lmdp.lmbda,
                s0=lmdp.s0,
                sparse_optimization=lmdp.sparse_optimization,
                verbose=lmdp.verbose,
                dtype=lmdp.dtype
            )
            self.P = lmdp.P
            self.R = lmdp.R
        
        self._print(f"Created LMDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
    
                  
    def _generate_R(self):
        """
        Generates the transition-based reward matrix (R) for the minigrid. 
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
            if self.environment.custom_grid.is_cliff(self.environment.custom_grid.state_index_mapper[j]) or self.environment.custom_grid.is_cliff(self.environment.custom_grid.state_index_mapper[i]):
                self.R[i, j] = self.dtype(-50)
            else:
                self.R[i, j] = self.dtype(-5)

        # Matrix R is now sparese as well, so if sparse_optimization is activated, we convert it.
        if self.sparse_optimization:
            self._print("Converting R into sparse matrix...")
            self._print(f"Memory usage before conversion: {getsizeof(self.R):,} bytes")
            self.R = csr_matrix(self.R)
            self._print(f"Memory usage after conversion: {getsizeof(self.R):,} bytes")
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None, show_window: bool = True) -> GameStats:
        """
        Visualizes the learnt policy on the MiniGrid environment.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): Custom policies to visualize. Each tuple is (step, policy). Defaults to None.
            num_times (int): Number of times to run the game. Defaults to 10.
            save_gif (bool): If True, saves the game sequence as a GIF. Defaults to False.
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.
            
            
        
        Returns:
            GameStats: An object containing statistics about the game played, such as number of moves, errors, and deaths.
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            self._print(f"Computing value function...")
            self.compute_value_function()
            return self.environment.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self, show_window=show_window, title="Power Iteration policy")
        else:
            assert policies is not None
            return self.environment.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self, show_window=show_window)

    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)

    
    def __eq__(self, obj):
        raise NotImplementedError("Method not implemented")