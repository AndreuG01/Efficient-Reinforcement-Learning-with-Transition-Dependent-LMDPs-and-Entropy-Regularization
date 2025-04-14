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
        max_steps: int = 200,
        allowed_actions: list[int] = None,
        **kwargs,
    ):
        
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        
        self.custom_grid = CustomGrid("minigrid", map=map, properties=properties, allowed_actions=allowed_actions)
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
    
    
    def visualize_policy(
        self,
        model: MinigridLMDP | MinigridMDP,
        policies: list[tuple[int, np.ndarray]],
        num_times: int=10,
        save_gif: bool = False,
        save_path: str = None,
    ) -> None:
        """
        Visualizes the behavior of the agent under some given policies by running multiple episodes, rendering each step, 
        and optionally saving the resulting frames as a GIF.

        Args:
            model (MinigridLMDP | MinigridMDP): The agent's model. It is used to determine the sequence of states that need to be visualized when following a policy.
            policies (list[tuple[int, np.ndarray]]): A list of policy arrays, one for each possible policy to visualize. Each policy contains the training epoch from which it was derived.
            num_times (int): The number of times to run each policy (default is 10).
            save_gif (bool): Whether to save the visualization as a GIF (default is False).
            save_path (str): The path to save the GIF if `save_gif` is True.
        
        Returns:
            None
        """
        frames = []
        if not save_gif:
            self.render_mode = "human"
        for policy_epoch, policy in policies:
            print(f"Visualizing policy from training epoch: {policy_epoch}")
            for i in tqdm(range(num_times), desc=f"Playing {num_times} games"):
                num_mistakes = 1
                self.reset()
                done = False
                actions = 0
                next_properties = {k: v[0] for k, v in self.custom_grid.state_properties.items()}
                
                next_layout = self.custom_grid.layout_combinations[0]
                while not done:
                    state = State(self.agent_pos[0], self.agent_pos[1], next_layout, **next_properties)
                    state_idx = next(k for k, v in self.custom_grid.state_index_mapper.items() if v == state)
                    
                    if isinstance(model, MinigridMDP):
                        if model.deterministic:
                            action = policy[state_idx]
                            
                        else:
                            next_state = np.random.choice(self.custom_grid.get_num_states(), p=model.P[state_idx, policy[state_idx], :])
                            if next_state != np.argmax(model.P[state_idx, policy[state_idx], :]):
                                print(f"MISTAKE {num_mistakes}")
                                num_mistakes += 1
                            # We need to get the action that leads to the next state
                            action = self.custom_grid.transition_action(state_idx, next_state, model.allowed_actions)
                            
                    else:
                        # next_state = np.argmax(policy[state_idx])
                        next_state = np.random.choice(self.custom_grid.get_num_states(), p=policy[state_idx].astype(np.float64))
                        if next_state != np.argmax(policy[state_idx]):
                            print(f"MISTAKE {num_mistakes}")
                            num_mistakes += 1
                        action = self.custom_grid.transition_action(state_idx, next_state, model.allowed_actions)
                    
                    next_state, _, terminal = self.custom_grid.move(state, action)
                    
                    next_properties = next_state.properties
                    next_layout = next_state.layout
                    
                        
                    frame = self.render()
                    if save_gif:
                        curr_frame = Image.fromarray(frame)
                        draw = ImageDraw.Draw(curr_frame)
                        
                        title_height = 40
                        frame_with_title = Image.new("RGB", (curr_frame.width, curr_frame.height + title_height), "white")
                        frame_with_title.paste(curr_frame, (0, title_height))
                        
                        # Add the text to the frame
                        draw_title = ImageDraw.Draw(frame_with_title)
                        title_text = f"Epoch: {policy_epoch}"
                        
                        font = ImageFont.load_default()
                        # font = ImageFont.truetype("UbuntuMono.ttf", size=20)  
                        text_bbox = draw_title.textbbox((0, 0), title_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        text_position = ((frame_with_title.width - text_width) // 2, (title_height - text_height) // 2)
                        draw_title.text(text_position, title_text, fill="black", font=font)
                        
                        frames.append(frame_with_title)
                    _, _, done, _, info = self.step(action)
                    
                    actions += 1
                    if actions == self.max_steps:
                        break
                    
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
        


class MinigridMDP(MDP):
    """
    A specialized Markov Decision Process (MDP) for MiniGrid environments.

    This class constructs an MDP representation from a MiniGrid-based map. It supports deterministic, stochastic,
    and mixed behaviour (stochastic for navigation and deterministic for manipulation) modes.

    Attributes:
        OFFSETS (dict[int, tuple[int, int]]): Maps direction indices to (dx, dy) movement.
        stochastic_prob (float): Probability of following the intended action in a stochastic setting.
        behaviour (Literal["deterministic", "stochastic", "mixed"]): Transition behaviour type. Defaults to "deterministic".
        num_actions (int): Number of allowed actions.
        allowed_actions (list[int]): List of action indices.
        minigrid_env (CustomMinigridEnv): MiniGrid environment wrapper.
        start_state (State): Initial state of the agent.
        num_states (int): The total numbe rof states (excluding terminal states).
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
        behaviour: Literal["deterministic", "stochastic", "mixed"] = "deterministic",
        benchmark_p: bool = False,
        threads: int = 4,
        mdp: MDP = None
    ):
        """
        Initializes a MinigridMDP instance.

        Args:
            map (Map): The map defining the MiniGrid layout.
            allowed_actions (list[int], optional): List of allowed action indices. If None, defaults to [0, 1, 2].
            properties (dict[str, list], optional): State properties for initialization. Defaults to {"orientation": [0, 1, 2, 3]}.
            stochastic_prob (float): Probability of following the intended action in a stochastic setting. Defaults to 0.9.
            behaviour (str): One of "deterministic", "stochastic", or "mixed". Defaults to "deterministic".
            mdp (MDP, optional): If provided, initializes this MinigridMDP using an existing MDP's parameters. Defaults to None.
        """
        
        self.stochastic_prob = stochastic_prob
        assert behaviour in ["deterministic", "stochastic", "mixed"], f"{behaviour} behaviour not supported."
        self.behaviour = behaviour
        deterministic = self.behaviour == "deterministic"
        
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.minigrid_env = CustomMinigridEnv(render_mode="rgb_array", map=map, properties=properties, allowed_actions=self.allowed_actions)
        start_pos = self.minigrid_env.custom_grid.start_pos
        self.start_state = [state for state in self.minigrid_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        
        print(f"MDP with {self.num_actions} actions. Allowed actions: {self.allowed_actions}")
        

        if mdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
                allowed_actions=self.allowed_actions,
                s0=self.minigrid_env.custom_grid.states.index(self.start_state),
                deterministic=deterministic,
                behaviour=self.behaviour
                # gamma=0.999
            )
            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.p_time = self.generate_P(
                    self.minigrid_env.custom_grid,
                    stochastic_prob=self.stochastic_prob,
                    benchmark=benchmark_p,
                    num_threads=threads
                )
            
            # If the agent has a mixed behaviour, we have to make navigation actions deterministic.
            if self.behaviour == "mixed":
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
            
            print(f"Created MDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
        else:
            # Useful when wanting to create a MinigridMDP from an embedding of an LMDP into an MDP
            # self.num_actions = mdp.num_actions
            # self.allowed_actions = [i for i in range(self.num_actions)]
            super().__init__(
                num_states=mdp.num_states,
                num_terminal_states=mdp.num_terminal_states,
                allowed_actions=self.allowed_actions,
                s0=mdp.s0,
                gamma=mdp.gamma,
                deterministic=mdp.deterministic,
                behaviour=self.behaviour
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
            state_repr = self.minigrid_env.custom_grid.states[state]
            if self.minigrid_env.custom_grid.is_cliff(state_repr):
                # For precision purposes, do not use rewards non strictily lower than np.log(np.finfo(np.float64).tiny) = -708
                self.R[state] = np.full(shape=self.num_actions, fill_value=-50, dtype=np.float64)
            else:
                self.R[state] = np.full(shape=self.num_actions, fill_value=-5, dtype=np.float64)

    
    def states_to_goal(self, include_actions: bool = False) -> list[int] | tuple[list[int], list[int]]:
        """
        Returns the sequence of state indices that lead to the goal according to the current policy.

        Args:
            include_actions (bool): If True, also returns the actions taken to make the transitions.

        Returns:
            list[int] or tuple: A list of state indices, or a tuple (states, actions).
        """
        curr_state = self.minigrid_env.custom_grid.state_index_mapper[self.s0]
        curr_state_idx = self.s0
        states = [self.s0]
        actions = []
        
        
        while not self.minigrid_env.custom_grid.is_terminal(curr_state):
            if self.behaviour == "stochastic":
                curr_state_idx = np.random.choice(self.minigrid_env.custom_grid.get_num_states(), p=self.P[curr_state_idx, self.policy[curr_state_idx], :])
                curr_state = self.minigrid_env.custom_grid.state_index_mapper[curr_state_idx]
            elif self.behaviour == "deterministic":
                curr_action = self.policy[curr_state_idx]
                curr_state, _, terminal = self.minigrid_env.custom_grid.move(self.minigrid_env.custom_grid.state_index_mapper[curr_state_idx], curr_action)
                if terminal:
                    curr_state_idx = self.minigrid_env.custom_grid.terminal_states.index(curr_state)
                else:
                    curr_state_idx = self.minigrid_env.custom_grid.states.index(curr_state)
            
                actions.append(curr_action)
            states.append(curr_state_idx)
        
        if include_actions:
            return (states, actions)
        
        return states


    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None) -> None:
        """
        Visualizes the learnt policy on the MiniGrid environment.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): Custom policies to visualize. Each tuple is (step, policy). Defaults to None.
            num_times (int): Number of times to run the game. Defaults to 10.
            save_gif (bool): If True, saves the game sequence as a GIF. Defaults to False.
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.
        
        Returns:
            None
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)
        else:
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)



class MinigridLMDP(LMDP):
    """
    A specialized linearly-solvable MDP for MiniGrid environments.

    This class constructs an LMDP representation from a MiniGrid-based map.
    
    Attributes:
        OFFSETS (dict[int, tuple[int, int]]): Maps direction indices to (dx, dy) movement.
        num_actions (int): Number of allowed actions.
        minigrid_env (CustomMinigridEnv): MiniGrid environment wrapper.
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
        lmdp: LMDP = None
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
        
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.minigrid_env = CustomMinigridEnv(render_mode="rgb_array", map=map, properties=properties, allowed_actions=self.allowed_actions)
        
        start_pos = self.minigrid_env.custom_grid.start_pos
        self.start_state = [state for state in self.minigrid_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        
        
        if lmdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
                s0=self.minigrid_env.custom_grid.states.index(self.start_state),
                # lmbda=0.99,
                sparse_optimization=sparse_optimization
            )

            if map.P is not None:
                assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
                self.P = map.P
            else:
                self.p_time = self.generate_P(
                    self.minigrid_env.custom_grid,
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
                    
        else:
            super().__init__(
                num_states=lmdp.num_states,
                num_terminal_states=lmdp.num_terminal_states,
                s0=lmdp.s0,
                lmbda=lmdp.lmbda,
                sparse_optimization=lmdp.sparse_optimization
            )
            self.P = lmdp.P
            self.R = lmdp.R

                  
    def _generate_R(self):
        """
        Generates the reward matrix (R) for the minigrid, setting the default reward to -50 for cliff states and to -5 for normal states.
        Terminal states get a reward of 0.

        Returns:
            None
        """
        self.R[:] = np.float64(-5)
        cliff_states = [i for i in range(self.num_states) if self.minigrid_env.custom_grid.is_cliff(self.minigrid_env.custom_grid.state_index_mapper[i])]
        self.R[cliff_states] = np.float64(-50)
        self.R[self.num_non_terminal_states:] = np.float64(0)

    
    def states_to_goal(self, stochastic: bool = False) -> list[int]:
        """
        Computes the trajectory of state indices from start to goal based on the current policy.

        Args:
            stochastic (bool): Whether to sample actions probabilistically or use greedy strategy. Defaults to False.

        Returns:
            list[int]: Sequence of state indices leading to the goal.
        """
        curr_state = self.minigrid_env.custom_grid.state_index_mapper[self.s0]
        curr_state_idx = self.s0
        states = [self.s0]
        
        while not self.minigrid_env.custom_grid.is_terminal(curr_state):        
            if stochastic:
                curr_state_idx = np.random.choice(self.minigrid_env.custom_grid.get_num_states(), p=self.policy[curr_state_idx])
            else:
                curr_state_idx = np.argmax(self.policy[curr_state_idx])
            
            curr_state = self.minigrid_env.custom_grid.state_index_mapper[curr_state_idx]
            states.append(curr_state_idx)
        
        return states
        
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None) -> None:
        """
        Visualizes the learnt policy on the MiniGrid environment.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): Custom policies to visualize. Each tuple is (step, policy). Defaults to None.
            num_times (int): Number of times to run the game. Defaults to 10.
            save_gif (bool): If True, saves the game sequence as a GIF. Defaults to False.
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.
        
        Returns:
            None
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)
        else:
            assert policies is not None
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)


class MinigridLMDP_TDR(LMDP_TDR):
    """
    A specialized LMDP with transition dependent-rewards for MiniGrid environments.

    This class constructs an LMDP representation from a MiniGrid-based map.
    
    Attributes:
        OFFSETS (dict[int, tuple[int, int]]): Maps direction indices to (dx, dy) movement.
        num_actions (int): Number of allowed actions.
        minigrid_env (CustomMinigridEnv): MiniGrid environment wrapper.
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
        threads: int = 4
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
        
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.minigrid_env = CustomMinigridEnv(render_mode="rgb_array", map=map, properties=properties, allowed_actions=self.allowed_actions)
        start_pos = self.minigrid_env.custom_grid.start_pos
        self.start_state = [state for state in self.minigrid_env.custom_grid.states if state.x == start_pos[1] and state.y == start_pos[0]][0]
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        
        super().__init__(
            self.num_states,
            num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
            s0=self.minigrid_env.custom_grid.states.index(self.start_state),
            sparse_optimization=sparse_optimization
        )

        if map.P is not None:
            assert map.P.shape == self.P.shape, f"Dimensions of custom transition probability function {map.P.shape} do not match the expected ones: {self.P.shape}"
            self.P = map.P
        else:
            self.p_time = self.generate_P(
                self.minigrid_env.custom_grid,
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
            if self.minigrid_env.custom_grid.is_cliff(self.minigrid_env.custom_grid.state_index_mapper[j]) or self.minigrid_env.custom_grid.is_cliff(self.minigrid_env.custom_grid.state_index_mapper[i]):
                self.R[i, j] = np.float64(-50)
            else:
                self.R[i, j] = np.float64(-5)

        # Matrix R is now sparese as well, so if sparse_optimization is activated, we convert it.
        if self.sparse_optimization:
            print("Converting R into sparse matrix...")
            print(f"Memory usage before conversion: {getsizeof(self.R):,} bytes")
            self.R = csr_matrix(self.R)
            print(f"Memory usage after conversion: {getsizeof(self.R):,} bytes")
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None) -> None:
        """
        Visualizes the learnt policy on the MiniGrid environment.

        Args:
            policies (list[tuple[int, np.ndarray]], optional): Custom policies to visualize. Each tuple is (step, policy). Defaults to None.
            num_times (int): Number of times to run the game. Defaults to 10.
            save_gif (bool): If True, saves the game sequence as a GIF. Defaults to False.
            save_path (str, optional): Path to save the GIF if `save_gif` is True. Defaults to None.
        
        Returns:
            None
        """
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)
        else:
            assert policies is not None
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)

    