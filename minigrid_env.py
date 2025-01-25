from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from grid import CustomGrid, CellType
from maps import Maps
from MDP import MDP
from gym.spaces import Discrete
import numpy as np
from PIL import Image
from algorithms import QLearning
import matplotlib.pyplot as plt
from state import State
from tqdm import tqdm

class MinigridActions:
    """
    Possible actions:
        Action number   |   Action type | Action description     |   Keyboard key     
        ------------------------------------------------------------------------
              0         |   left        |   Turn left            |   Left
              1         |   right       |   Turn right           |   Right
              2         |   forward     |   Move forward         |   Up
              3         |   pikup       |   Pickup an object     |   Pageup / Tab
              4         |   drop        |   Drop an object       |   Pagedown / Left shift
              5         |   toggle      |   Toggle               |   Space
              6         |   done        |   Done                 |   Enter
    """
    ROTATE_LEFT = 0
    ROTATE_RIGHT = 1
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6
    


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
    - custom_grid (CustomGrid): A grid representation with different cell types and properties.
    - agent_start_pos (tuple[int, int]): The starting position of the agent in the grid.
    - agent_start_dir (int): The starting direction (orientation) of the agent. Can be one of [0, 1, 2, 3] representing right, down, left, and up respectively.
    - num_directions (int): The total number of possible directions the agent can face (default is 4: right, down, left, up).
    - mission_space (MissionSpace): Defines the mission and possible tasks for the agent.
    - max_steps (int): The maximum number of steps allowed for an episode.
    """
    def __init__(
        self,
        properties: dict[str, list] = None,
        map:list[str] = None,
        grid_size: int = 3,
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        
        self.num_directions = 4
        self.custom_grid = CustomGrid(map=map, grid_size=grid_size, properties=properties)
        self.agent_start_pos = self.custom_grid.start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * grid_size ** 3

        super().__init__(
            mission_space=mission_space,
            # grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            width=self.custom_grid.size_y,
            height=self.custom_grid.size_x,
            **kwargs,
        )
        

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate the walls
        for state in self.custom_grid.positions[CellType.WALL]:
            self.grid.set(state.y, state.x, Wall())
        
        first_goal = self.custom_grid.goal_pos[0]
        self.put_obj(Goal(), first_goal[1], first_goal[0])
        for goal_state in self.custom_grid.goal_pos:
            if goal_state[0] != first_goal[0] and goal_state[1] != first_goal[1]:
                self.put_obj(Goal(), goal_state[1], goal_state[0])
        
        for cliff_state in self.custom_grid.positions[CellType.CLIFF]:
            self.put_obj(Lava(), cliff_state.y, cliff_state.x)
        
        # # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        # if self.agent_start_pos is not None:
        #     self.agent_pos = self.agent_start_pos
        #     self.agent_dir = self.agent_start_dir
        # else:
        #     self.place_agent()

        self.mission = "grand mission"
    
        
    
    
    def visualize_policy(self, policies: list[np.ndarray], num_times: int=10, save_gif: bool = False, save_path: str = None):
        """
        Visualizes the behavior of the agent under some given policies by running multiple episodes, rendering each step, 
        and optionally saving the resulting frames as a GIF.

        Args:
        - policies (list[np.ndarray]): A list of policy arrays, one for each possible policy to visualize.
        - num_times (int): The number of times to run each policy (default is 10).
        - save_gif (bool): Whether to save the visualization as a GIF (default is False).
        - save_path (str): The path to save the GIF if `save_gif` is True.
        """
        frames = []
        if not save_gif:
            self.render_mode = "human"
        for policy in policies:
            for i in tqdm(range(num_times), desc=f"Playing {num_times} games"):
                self.reset()
                done = False
                while not done:
                    state = State(self.agent_pos[0], self.agent_pos[1], **{"orientation": self.agent_dir})
                    state_idx = next(k for k, v in self.custom_grid.state_index_mapper.items() if v == state)
                    
                    action = policy[state_idx]
                    frame = self.render()
                    if save_gif: frames.append(Image.fromarray(frame))
                    _, _, done, _, _ = self.step(action)
        
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
    
    OFFSETS = {
        0: (1, 0),   # RIGHT
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (0, -1)   # UP
    }
    
    def __init__(self, grid_size: int = 3, map: list[str] = None, allowed_actions: list[int] = None, deterministic: bool = True, properties: dict[str, list] = {"orientation": [i for i in range(4)]}):
        
        self.deterministic = deterministic
        
        self.minigrid_env = CustomMinigridEnv(grid_size=grid_size, render_mode="rgb_array", map=map, properties=properties)
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        self.num_actions = len(allowed_actions)
        
        
        start_pos = self.minigrid_env.custom_grid.start_pos
        
        
        super().__init__(
            self.num_states,
            num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
            num_actions=self.num_actions,
            s0=self.minigrid_env.custom_grid.positions[CellType.NORMAL].index(State(start_pos[0], start_pos[1], **{k: v[0] for k, v in self.minigrid_env.custom_grid.state_properties.items()}))
        )

        # self._generate_P()
        self.generate_P(self.minigrid_env.custom_grid.positions, self.move, self.minigrid_env.custom_grid)
        self._generate_R()
                
    
    def move(self, state: State, action: int):
        # TODO: update when more actions are added
        orientation = state.properties["orientation"]
        y = state.y
        x = state.x
        
        next_state = State(y, x, **state.properties)
        
        if action == MinigridActions.ROTATE_LEFT:
            next_state.properties["orientation"] = (orientation - 1) % 4
            
            return next_state, True, False
        
        elif action == MinigridActions.ROTATE_RIGHT:
            next_state.properties["orientation"] = (orientation + 1) % 4
            return next_state, True, False
        
        elif action == MinigridActions.FORWARD:
            dy, dx = self.OFFSETS[orientation]
            next_state.y = y + dy
            next_state.x = x + dx
            in_bounds = self.minigrid_env.custom_grid.is_valid(next_state)
            if not in_bounds: next_state = state
        
            return next_state, in_bounds, self.minigrid_env.custom_grid.is_terminal(next_state)

                
                  
    def _generate_R(self):
        pos = self.minigrid_env.custom_grid.positions
        for j in range(self.minigrid_env.custom_grid.size_x):
            for i in range(self.minigrid_env.custom_grid.size_y):
                for orientation in range(self.minigrid_env.num_directions):
                    tmp_state = State(i, j, **{"orientation": orientation})
                    if tmp_state in pos[CellType.NORMAL]:
                        self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.int32)
                    if tmp_state in pos[CellType.CLIFF]:
                        self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-10, dtype=np.int32)



    def visualize_policy(self, policies: list[np.ndarray] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None):
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy([self.policy], num_times=num_times, save_gif=save_gif, save_path=save_path)
        else:
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path)
            