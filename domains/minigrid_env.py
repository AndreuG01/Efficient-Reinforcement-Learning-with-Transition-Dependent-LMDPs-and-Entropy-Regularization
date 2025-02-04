from __future__ import annotations

from copy import deepcopy
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
from utils.state import State, Object
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
    
    @classmethod
    def get_actions(cls) -> list[int]:
        return [value for key, value in cls.__dict__.items() if type(value) == int]


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
        objects: list[Object] = None,
        **kwargs,
    ):
        
        self.num_directions = 4
        self.custom_grid = CustomGrid(map=map, grid_size=grid_size, properties=properties, objects=objects)
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
        for object in self.custom_grid.objects:
            if object.type == "door":
                self.grid.set(object.x, object.y, Door(object.color, is_locked=True))
            elif object.type == "key":
                self.grid.set(object.x, object.y, Key(object.color))
        # if len(self.custom_grid.positions[CellType.DOOR]) > 0: # There are doors in the environment
        #     first_door = self.custom_grid.positions[CellType.DOOR][0]
        #     color = 1
        #     for door_state in self.custom_grid.positions[CellType.DOOR]:
        #         if door_state.y != first_door.y and door_state.x != first_door.x:
        #             self.grid.set(door_state.y, door_state.x, Door(COLOR_NAMES[color], is_locked=True))
        #             color += 1
        
        # if len(self.custom_grid.positions[CellType.KEY]) > 0: # There are keys in the environment
        #     first_key = self.custom_grid.positions[CellType.KEY][0]
        #     color = 1
        #     for key_state in self.custom_grid.positions[CellType.KEY]:
        #         if key_state.y != first_key.y and key_state.x != first_key.x:
        #             self.grid.set(key_state.y, key_state.x, Key(COLOR_NAMES[color]))
        #             color += 1            
        
        
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
        model: MinigridLMDP | MinigridMDP,
        policies: list[tuple[int, np.ndarray]],
        num_times: int=10,
        save_gif: bool = False,
        save_path: str = None,
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
                next_properties = {k: v[0] for k, v in self.custom_grid.state_properties.items()}
                
                next_layout = self.custom_grid.layout_combinations[0]
                # print(next_layout)
                while not done:
                    state = State(self.agent_pos[0], self.agent_pos[1], next_layout, **next_properties)
                    # print(state)
                    state_idx = next(k for k, v in self.custom_grid.state_index_mapper.items() if v == state)
                    
                    if isinstance(model, MinigridMDP):
                        if model.deterministic:
                            action = policy[state_idx]
                            # action = model.action_to_action[state_idx][policy[state_idx]]
                        else:
                            next_state = np.random.choice(self.custom_grid.get_num_states(), p=model.P[state_idx, policy[state_idx], :])
                            # next_state = np.argmax(model.P[state_idx, policy[state_idx]])
                            # We need to get the action that leads to the next state
                            # print(p[state_idx, action, :])
                            action = model.transition_action(state_idx, next_state, )
                            
                    else:
                        next_state = policy[state_idx]
                        action = model.transition_action(state_idx, next_state)
                    print(action)
                    if action == MinigridActions.DROP:
                        print("need to drop")
                    if action == MinigridActions.DROP: action += 1 # TODO: remove is DROP action is sometimes used
                    if action == MinigridActions.TOGGLE:
                        door_opened = not door_opened
                    
                    next_state, _, _ = model.move(state, action, modify=True)
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
    
    OFFSETS = {
        0: (1, 0),   # RIGHT
        1: (0, 1),   # DOWN
        2: (-1, 0),  # LEFT
        3: (0, -1)   # UP
    }
    
    def __init__(
        self,
        grid_size: int = 3,
        map: list[str] = None,
        allowed_actions: list[int] = None,
        deterministic: bool = True,
        properties: dict[str, list] = {"orientation": [i for i in range(4)]},
        objects: list[Object] = None,
        mdp: MDP = None
    ):
        
        self.deterministic = deterministic
        
        self.minigrid_env = CustomMinigridEnv(grid_size=grid_size, render_mode="human", map=map, properties=properties, objects=objects)
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        print(self.num_states)
        self.num_actions = len(allowed_actions)
        self.allowed_actions = allowed_actions
        
        
        start_pos = self.minigrid_env.custom_grid.start_pos
        
        
    
        if mdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
                allowed_actions=allowed_actions,
                s0=self.minigrid_env.custom_grid.positions[CellType.NORMAL].index(State(start_pos[0], start_pos[1], self.minigrid_env.custom_grid.layout_combinations[0], **{k: v[0] for k, v in self.minigrid_env.custom_grid.state_properties.items()})),
                gamma=0.8
            )

            self.generate_P(self.minigrid_env.custom_grid.positions, self.move, self.minigrid_env.custom_grid)
            self._generate_R()
        else:
            # Useful when wanting to create a MinigridMDP from an embedding of an LMDP into an MDP
            super().__init__(
                num_states=mdp.num_states,
                num_terminal_states=mdp.num_terminal_states,
                allowed_actions=[i for i in range(self.num_actions)],
                s0=mdp.s0
            )
            
            self.P = mdp.P
            self.R = mdp.R
                
    
    
    
    def move(self, state: State, action: int, modify: bool = False):
        # TODO: update when more actions are added
        orientation = state.properties["orientation"]
        y = state.y
        x = state.x
        curr_layout = state.layout
    
        # if action == MinigridActions.DROP: action += 1
        next_state = State(y, x, layout=deepcopy(state.layout), **state.properties)
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
            if (new_x, new_y) in curr_layout: next_object = curr_layout[(new_x, new_y)]
            else: next_object = None
            
            if next_object is not None and next_object.type == "key" and not state.properties[f"{next_object.color}_key_{next_object.id}"]:
                # If there is a key in the next state, the agent remains at the same state.
                in_bounds = True
            
            elif next_object is not None and next_object.type == "door" and not state.properties[f"{next_object.color}_door_{next_object.id}"]:
                # If in the next state there is a door and it is not opened, the agent remains where it is
                in_bounds = True
            
            else: #if not (self.minigrid_env.custom_grid.is_key(State(new_y, new_x, object=next_object, **state.properties)) and self.minigrid_env.custom_grid.is_door(State(new_y, new_x, object=next_object, **state.properties))):
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
            if (new_x, new_y) in curr_layout: next_object = curr_layout[(new_x, new_y)]
            else:
                next_object = None
                return next_state, True, False
            
            if next_object is not None and next_object.type == "key" and not self.minigrid_env.custom_grid.state_has_key(state) and (len([object for object in curr_layout.values() if object is not None and object.type == "key"]) == len([obj for obj in self.minigrid_env.custom_grid.objects if obj.type == "key"])):
                next_state.properties[f"{next_object.color}_key_{next_object.id}"] = True # The agent picks up the key
                next_state.layout[(new_x, new_y)] = None
            
            return next_state, True, False
            
        elif action == MinigridActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the dorr is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            dy, dx = self.OFFSETS[orientation]
            new_y = y + dy
            new_x = x + dx
            
            if (new_x, new_y) in curr_layout: next_object = curr_layout[(new_x, new_y)]
            else: next_object = None
            
            if next_object is not None and next_object.type == "door" and self.minigrid_env.custom_grid.state_has_key_color(state, next_object.color):
                next_state.properties[f"{next_object.color}_door_{next_object.id}"] = not next_state.properties[f"{next_object.color}_door_{next_object.id}"]
            return next_state, True, False
        
        # elif action == MinigridActions.DROP:
        #     # If the agent is wearing a key and the position towards which it is facing is an empty square, then the agent can drop the object
        #     dy, dx = self.OFFSETS[orientation]
        #     new_y = y + dy
        #     new_x = x + dx
        #     if (new_x, new_y) in curr_layout:
        #         next_object = curr_layout[(new_x, new_y)]
        #     else:
        #         next_object = None
        #         return next_state, True, False
           
        #     if next_object is None and self.minigrid_env.custom_grid.state_has_key(state) and self.minigrid_env.custom_grid.is_normal(State(new_y, new_x, curr_layout, **state.properties)) and not any([object is not None and "key" in object.type for object in list(curr_layout.values())]):
            
        #         carrying_object = self.minigrid_env.custom_grid.get_carrying_object(state)
        #         next_state.layout[(new_x, new_y)] = carrying_object
        #         # self.minigrid_env.custom_grid.set_state_object_visibility(new_x, new_y, carrying_object, True)
            
                
        #         next_state.properties = {k: v if "key" not in k else False for k, v in next_state.properties.items()}
            
        #     return next_state, True, False
        else:
            # Drop and Done actions have no effect yet
            return next_state, True, False
                  
    def _generate_R(self):
        properties_combinations = [elem for elem in self.minigrid_env.custom_grid._get_property_combinations()]
        
        pos = self.minigrid_env.custom_grid.positions
        for state in range(self.num_non_terminal_states):
            if self.minigrid_env.custom_grid.state_index_mapper[state] in pos[CellType.CLIFF]:
                self.R[state] = np.full(shape=self.num_actions, fill_value=-10, dtype=np.int32)
            else:
                self.R[state] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.int32)
                
        # for j in range(self.minigrid_env.custom_grid.size_x):
        #     for i in range(self.minigrid_env.custom_grid.size_y):
        #         for vals in properties_combinations:
        #             tmp_state = State(i, j, **dict(zip(list(self.minigrid_env.custom_grid.state_properties.keys()), vals)))
        #             if tmp_state in pos[CellType.NORMAL]:
        #                 self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.int32)
        #             if tmp_state in pos[CellType.CLIFF]:
        #                 self.R[pos[CellType.NORMAL].index(tmp_state)] = np.full(shape=self.num_actions, fill_value=-10, dtype=np.int32)


    def transition_action(self, state, next_state):
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
            self.minigrid_env.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)
        else:
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)



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
        
        if "blue_door" in state.properties:
            door_opened = state.properties["blue_door"]
        if "blue_key" in state.properties:
            has_key = state.properties["blue_key"]
        
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
                            if "blue_door" in self.minigrid_env.custom_grid.state_properties:
                                tmp_state = State(i, j, **{"orientation": orientation, "blue_key": key, "blue_door": door})
                            else:
                                tmp_state = State(i, j, **{"orientation": orientation})
                            if tmp_state in pos[CellType.NORMAL]:
                                self.R[pos[CellType.NORMAL].index(tmp_state)] = -1
                            if tmp_state in pos[CellType.CLIFF]:
                                self.R[pos[CellType.NORMAL].index(tmp_state)] = -10



    def transition_action(self, state, next_state):
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
            self.minigrid_env.visualize_policy([[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, mdp=False, transition_action=self.transition_action)
        else:
            assert policies is not None
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, mdp=False, transition_action=self.transition_action)

    