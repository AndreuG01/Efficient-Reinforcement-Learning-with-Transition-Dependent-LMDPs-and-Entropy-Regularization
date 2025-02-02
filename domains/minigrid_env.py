from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from .grid import CustomGrid, CellType
from models.MDP import MDP
from models.LMDP import LMDP
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.state import State
from tqdm import tqdm
from collections.abc import Callable

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
        self.max_steps = max_steps
        if max_steps is None:
            max_steps = 70

        super().__init__(
            mission_space=mission_space,
            # grid_size=size,
            highlight=False, # To avoid seeing the agent's view, which is not considered in our models.
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
        
        # TODO: modify to acount for the correct key-door mapping
        first_door = self.custom_grid.positions[CellType.DOOR][0]
        self.grid.set(first_door.y, first_door.x, Door(COLOR_NAMES[0], is_locked=True))
        color = 1
        for door_state in self.custom_grid.positions[CellType.DOOR]:
            if door_state.y != first_door.y and door_state.x != first_door.x:
                self.grid.set(door_state.y, door_state.x, Door(COLOR_NAMES[color], is_locked=True))
                color += 1
        
        first_key = self.custom_grid.positions[CellType.KEY][0]
        self.grid.set(first_key.y, first_key.x, Key(COLOR_NAMES[0]))
        color = 1
        for key_state in self.custom_grid.positions[CellType.KEY]:
            if key_state.y != first_key.y and key_state.x != first_key.x:
                self.grid.set(key_state.y, key_state.x, Key(COLOR_NAMES[color]))
                color += 1            
        
        
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
    
        
    
    
    def visualize_policy(
        self,
        policies: list[tuple[int, np.ndarray]],
        num_times: int=10,
        save_gif: bool = False,
        save_path: str = None,
        mdp: bool = False,
        policy_to_action: Callable = None
    ):
        """
        Visualizes the behavior of the agent under some given policies by running multiple episodes, rendering each step, 
        and optionally saving the resulting frames as a GIF.

        Args:
        - policies (list[tuple[int, np.ndarray]]): A list of policy arrays, one for each possible policy to visualize. Each policy contains the training epoch from which it was derived.
        - num_times (int): The number of times to run each policy (default is 10).
        - save_gif (bool): Whether to save the visualization as a GIF (default is False).
        - save_path (str): The path to save the GIF if `save_gif` is True.
        """
        frames = []
        if not save_gif:
            self.render_mode = "human"
        for policy_epoch, policy in policies:
            print(f"Visualizing policy from training epoch: {policy_epoch}")
            for i in tqdm(range(num_times), desc=f"Playing {num_times} games"):
                self.reset()
                done = False
                actions = 0
                has_key = False
                door_opened = False
                while not done:
                    
                    state = State(self.agent_pos[0], self.agent_pos[1], **{"orientation": self.agent_dir, "blue_key": has_key, "blue_door": door_opened})
                    state_idx = next(k for k, v in self.custom_grid.state_index_mapper.items() if v == state)
                    
                    if mdp:
                        action = policy[state_idx]
                    else:
                        next_state = policy[state_idx]
                        action = policy_to_action(state_idx, next_state)
                        
                    if action == MinigridActions.TOGGLE:
                        # action += 1
                        door_opened = not door_opened
                    if action == MinigridActions.PICKUP:
                        has_key = True
                        
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
            allowed_actions=allowed_actions,
            s0=self.minigrid_env.custom_grid.positions[CellType.NORMAL].index(State(start_pos[0], start_pos[1], **{k: v[0] for k, v in self.minigrid_env.custom_grid.state_properties.items()}))
        )

        self.generate_P(self.minigrid_env.custom_grid.positions, self.move, self.minigrid_env.custom_grid)
        self._generate_R()
                
    
    def move(self, state: State, action: int):
        # TODO: update when more actions are added
        orientation = state.properties["orientation"]
        door_opened = state.properties["blue_door"]
        has_key = state.properties["blue_key"]
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
            new_y = y + dy
            new_x = x + dx
            
            if self.minigrid_env.custom_grid.is_key(State(new_y, new_x, **state.properties)) and not has_key:
                # If there is a key in the next state and the agent does not have it yet, the agent remains at the same state.
                in_bounds = True
            
            elif self.minigrid_env.custom_grid.is_door(State(new_y, new_x, **state.properties)) and not door_opened:
                # If in the next state there is a door and it is not opened, the agent remains where it is
                in_bounds = True
            
            elif not (self.minigrid_env.custom_grid.is_key(State(new_y, new_x, **state.properties)) and self.minigrid_env.custom_grid.is_door(State(new_y, new_x, **state.properties))):
                # The agent moves as usually
                next_state.y = y + dy
                next_state.x = x + dx
                in_bounds = self.minigrid_env.custom_grid.is_valid(next_state)
                if not in_bounds: next_state = state
            
            
        
            return next_state, in_bounds, self.minigrid_env.custom_grid.is_terminal(next_state)

        elif action == MinigridActions.PICKUP:
            # If the agent is facing a key, it gets it. Otherwise, it remains at the same state
            dy, dx = self.OFFSETS[orientation]
            new_y = y + dy
            new_x = x + dx
            if self.minigrid_env.custom_grid.is_key(State(new_y, new_x, **state.properties)):
                next_state.properties["blue_key"] = True # The agent picks up the key
            
            return next_state, True, False
            
        elif action == MinigridActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the dorr is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            dy, dx = self.OFFSETS[orientation]
            new_y = y + dy
            new_x = x + dx
            if self.minigrid_env.custom_grid.is_door(State(new_y, new_x, **state.properties)) and has_key:
                next_state.properties["blue_door"] = not door_opened
            
            return next_state, True, False
                    
        else:
            # Drop and Done actions have no effect yet
            return next_state, True, False
                  
    def _generate_R(self):
        pos = self.minigrid_env.custom_grid.positions
        for j in range(self.minigrid_env.custom_grid.size_x):
            for i in range(self.minigrid_env.custom_grid.size_y):
                for orientation in range(self.minigrid_env.num_directions):
                    for door in [False, True]:
                        for key in [False, True]:
                            tmp_state = State(i, j, **{"orientation": orientation, "blue_key": key, "blue_door": door})
                            if tmp_state in pos[CellType.NORMAL]:
                                self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.int32)
                            if tmp_state in pos[CellType.CLIFF]:
                                self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-10, dtype=np.int32)



    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None):
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy([[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, mdp=True)
        else:
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, mdp=True)



class MinigridLMDP(LMDP):
    
    OFFSETS = {
        0: (1, 0),   # RIGHT
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (0, -1)   # UP
    }
    
    def __init__(self, grid_size: int = 3, map: list[str] = None, allowed_actions: list[int] = None, properties: dict[str, list] = {"orientation": [i for i in range(4)]}):
        
        self.minigrid_env = CustomMinigridEnv(grid_size=grid_size, render_mode="rgb_array", map=map, properties=properties)
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        self.num_actions = len(allowed_actions)
        self.allowed_actions = allowed_actions
        
        start_pos = self.minigrid_env.custom_grid.start_pos
        
        
        super().__init__(
            self.num_states,
            num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
            s0=self.minigrid_env.custom_grid.positions[CellType.NORMAL].index(State(start_pos[0], start_pos[1], **{k: v[0] for k, v in self.minigrid_env.custom_grid.state_properties.items()}))
        )

        self.generate_P(self.minigrid_env.custom_grid.positions, self.move, self.minigrid_env.custom_grid, self.allowed_actions)
        self._generate_R()
                
    
    # TODO: same as MinigridMDP
    def move(self, state: State, action: int):
        # TODO: update when more actions are added
        orientation = state.properties["orientation"]
        door_opened = state.properties["blue_door"]
        has_key = state.properties["blue_key"]
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
            new_y = y + dy
            new_x = x + dx
            
            if self.minigrid_env.custom_grid.is_key(State(new_y, new_x, **state.properties)) and not has_key:
                # If there is a key in the next state and the agent does not have it yet, the agent remains at the same state.
                in_bounds = True
            
            elif self.minigrid_env.custom_grid.is_door(State(new_y, new_x, **state.properties)) and not door_opened:
                # If in the next state there is a door and it is not opened, the agent remains where it is
                in_bounds = True
            
            elif not (self.minigrid_env.custom_grid.is_key(State(new_y, new_x, **state.properties)) and self.minigrid_env.custom_grid.is_door(State(new_y, new_x, **state.properties))):
                # The agent moves as usually
                next_state.y = y + dy
                next_state.x = x + dx
                in_bounds = self.minigrid_env.custom_grid.is_valid(next_state)
                if not in_bounds: next_state = state
            
            
        
            return next_state, in_bounds, self.minigrid_env.custom_grid.is_terminal(next_state)

        elif action == MinigridActions.PICKUP:
            # If the agent is facing a key, it gets it. Otherwise, it remains at the same state
            dy, dx = self.OFFSETS[orientation]
            new_y = y + dy
            new_x = x + dx
            if self.minigrid_env.custom_grid.is_key(State(new_y, new_x, **state.properties)):
                next_state.properties["blue_key"] = True # The agent picks up the key
            
            return next_state, True, False
            
        elif action == MinigridActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the dorr is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            dy, dx = self.OFFSETS[orientation]
            new_y = y + dy
            new_x = x + dx
            if self.minigrid_env.custom_grid.is_door(State(new_y, new_x, **state.properties)) and has_key:
                next_state.properties["blue_door"] = not door_opened
            
            return next_state, True, False
                    
        else:
            # Drop and Done actions have no effect yet
            return next_state, True, False
                  
    def _generate_R(self):
        pos = self.minigrid_env.custom_grid.positions
        for j in range(self.minigrid_env.custom_grid.size_x):
            for i in range(self.minigrid_env.custom_grid.size_y):
                for orientation in range(self.minigrid_env.num_directions):
                    for door in [False, True]:
                        for key in [False, True]:
                            tmp_state = State(i, j, **{"orientation": orientation, "blue_key": key, "blue_door": door})
                            if tmp_state in pos[CellType.NORMAL]:
                                self.R[pos[CellType.NORMAL].index(tmp_state)] = -1
                            if tmp_state in pos[CellType.CLIFF]:
                                self.R[pos[CellType.NORMAL].index(tmp_state)] = -10



    def policy_to_action(self, state, next_state):
        curr_state = self.minigrid_env.custom_grid.state_index_mapper[state]
        
        for action in self.allowed_actions:
            move_state, _, _ = self.move(curr_state, action)
            # move_idx = next([k for k, v in self.minigrid_env.custom_grid.state_index_mapper.items() if v == move_state])
            if move_state == self.minigrid_env.custom_grid.state_index_mapper[next_state]:
                return action
        
        return 0
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None):
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy([[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, mdp=False, policy_to_action=self.policy_to_action)
        else:
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, mdp=False, policy_to_action=self.policy_to_action)

    