from __future__ import annotations

from copy import deepcopy
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.minigrid_env import MiniGridEnv
from .grid import CustomGrid, CellType
from models.MDP import MDP
from models.LMDP import LMDP
from models.LMDP_TDR import LMDP_TDR
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.state import State, Object
from tqdm import tqdm
from collections.abc import Callable
from scipy.sparse import csr_matrix
from sys import getsizeof

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
            max_steps = 200
            # max_steps = self.custom_grid.get_num_states() // 3

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
                            # We need to get the action that leads to the next state
                            action = model.transition_action(state_idx, next_state)
                            # print(f"Action chosen at move {actions}: {action}")
                            # print(f"Next_state idx {next_state}, {np.where(model.P[state_idx, policy[state_idx], :] != 0)}, {model.P[state_idx, policy[state_idx], np.where(model.P[state_idx, policy[state_idx], :] != 0)[0]]}")
                            
                    else:
                        next_state = np.random.choice(self.custom_grid.get_num_states(), p=policy[state_idx])
                        print(f"State: {state_idx}, probs: {policy[state_idx]}")
                        action = model.transition_action(state_idx, next_state)
                    
                    next_state, _, _ = model.move(state, action)
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
        stochastic_prob: float = 0.9,
        mdp: MDP = None
    ):
        
        self.stochastic_prob = stochastic_prob
        self.deterministic = deterministic
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.minigrid_env = CustomMinigridEnv(grid_size=grid_size, render_mode="rgb_array", map=map, properties=properties, objects=objects)
        self.remove_unreachable_states()
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        
        print(f"MDP with {self.num_actions} actions. Allowed actions: {self.allowed_actions}")
        
        
        start_pos = self.minigrid_env.custom_grid.start_pos
        
        
    
        if mdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
                allowed_actions=self.allowed_actions,
                s0=0,
                deterministic=self.deterministic
                # gamma=0.999
            )

            self.generate_P(self.move, self.minigrid_env.custom_grid, stochastic_prob=self.stochastic_prob)
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
                deterministic=mdp.deterministic
            )
            
            self.P = mdp.P
            self.R = mdp.R
    
    # TODO: generalize, as this is shared among all MDP, LMDP and LMDP_TDR
    def remove_unreachable_states(self):
        print("Going to remove unreachable states")
        
        reachable_states = set()
        queue = list(self.minigrid_env.custom_grid.terminal_states)

        for terminal_state in queue:
            reachable_states.add(terminal_state)

        while queue:
            current_state = queue.pop(0)
            for action in self.allowed_actions:
                next_state, _, _ = self.move(current_state, action)
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    queue.append(next_state)

        states = [state for state in self.minigrid_env.custom_grid.states if state in reachable_states]
        terminal_states = [state for state in self.minigrid_env.custom_grid.terminal_states if state in reachable_states]

        removed_states = len(self.minigrid_env.custom_grid.states) - len(states)
        print(f"Removing {removed_states} states")

        self.minigrid_env.custom_grid.states = states
        self.minigrid_env.custom_grid.terminal_states = terminal_states
        self.minigrid_env.custom_grid.generate_state_index_mapper()
        
    
    def move(self, state: State, action: int):
        orientation = state.properties["orientation"]
        y, x, curr_layout = state.y, state.x, state.layout
        
        next_state = State(y, x, layout=deepcopy(state.layout), **state.properties)
        
        if action in [MinigridActions.ROTATE_LEFT, MinigridActions.ROTATE_RIGHT]:
            next_state.properties["orientation"] = (orientation + (1 if action == MinigridActions.ROTATE_RIGHT else -1)) % 4
            
            return next_state, True, False
        
        dy, dx = self.OFFSETS[orientation]
        new_y, new_x = y + dy, x + dx
        next_object = curr_layout.get((new_x, new_y)) if curr_layout else None
        
        if action == MinigridActions.FORWARD:
            if next_object is not None and next_object.type == "key":
                # If there is a key in the next state, the agent remains at the same state.
                in_bounds = True
            
            elif next_object is not None and next_object.type == "door" and not state.properties[f"{next_object.color}_door_{next_object.id}"]:
                # If in the next state there is a door and it is not opened, the agent remains where it is
                in_bounds = True
            
            else:
                # The agent moves as usually
                next_state.y = y + dy
                next_state.x = x + dx
                in_bounds = self.minigrid_env.custom_grid.is_valid(next_state)
                if not in_bounds: next_state = state
        
            return next_state, in_bounds, self.minigrid_env.custom_grid.is_terminal(next_state)

        elif action == MinigridActions.PICKUP:
            # If the agent is facing a key, it gets it. Otherwise, it remains at the same state
            if not curr_layout: return next_state, True, False
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.minigrid_env.custom_grid.get_num_keys() - 1
            if next_object is not None and next_object.type == "key" and not agent_has_key:
                # print("pickup")
                next_state.layout[(new_x, new_y)] = None
            
            return next_state, True, False
            
        elif action == MinigridActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the door is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            if not curr_layout: return next_state, True, False
            missing_key = [obj for obj in self.minigrid_env.custom_grid.objects if obj not in curr_layout.values()]
            if len(missing_key) == 0:
                return next_state, True, False
            
            missing_key: Object = missing_key[0]
            
            if next_object is not None and next_object.type == "door" and missing_key.color == next_object.color:
                # print("toggle")
                next_state.properties[f"{next_object.color}_door_{next_object.id}"] = not next_state.properties[f"{next_object.color}_door_{next_object.id}"]
            return next_state, True, False
        
        elif action == MinigridActions.DROP:
            # If the agent is wearing a key and the position towards which it is facing is an empty square, then the agent can drop the object
            if not curr_layout: return next_state, True, False
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.minigrid_env.custom_grid.get_num_keys() - 1
           
            if next_object is None and agent_has_key and self.minigrid_env.custom_grid.is_normal(State(new_y, new_x, curr_layout, **state.properties)):
                carrying_object = [obj for obj in self.minigrid_env.custom_grid.objects if obj.type == "key" and obj not in layout_keys][0]
            
                next_state.layout[(new_x, new_y)] = carrying_object
            
            return next_state, True, False
        else:
            # Done actions have no effect yet
            return next_state, True, False

          
    def _generate_R(self):
        for state in range(self.num_non_terminal_states):
            state_repr = self.minigrid_env.custom_grid.states[state]
            if self.minigrid_env.custom_grid.is_cliff(state_repr):
                # For precision purposes, do not use rewards non strictily lower than np.log(np.finfo(np.float64).tiny) = -708
                self.R[state] = np.full(shape=self.num_actions, fill_value=-10, dtype=np.float64)
            else:
                self.R[state] = np.full(shape=self.num_actions, fill_value=-1, dtype=np.float64)



    def transition_action(self, state_idx, next_state_idx):
        curr_state = self.minigrid_env.custom_grid.state_index_mapper[state_idx]
        for action in self.allowed_actions:
            move_state, _, _ = self.move(curr_state, action)
            next_state = self.minigrid_env.custom_grid.state_index_mapper[next_state_idx]
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
        # if not hasattr(self, "V") or self.V is None and policies is None:
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
    
    def __init__(
        self,
        grid_size: int = 3,
        map: list[str] = None,
        allowed_actions: list[int] = None,
        properties: dict[str, list] = {"orientation": [i for i in range(4)]},
        objects: list[Object] = None,
        sparse_optimization: bool = True,
        benchmark_p: bool = False,
        threads: int = 4,
        lmdp: LMDP = None
    ):
        
        if allowed_actions:
            self.num_actions = len(allowed_actions)
            self.allowed_actions = allowed_actions
        else:
            self.num_actions = 3
            self.allowed_actions = [i for i in range(self.num_actions)]
        
        self.minigrid_env = CustomMinigridEnv(grid_size=grid_size, render_mode="rgb_array", map=map, properties=properties, objects=objects)
        self.remove_unreachable_states()
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        
        start_pos = self.minigrid_env.custom_grid.start_pos
        
        
        if lmdp is None:
            super().__init__(
                self.num_states,
                num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
                s0=0,
                # lmbda=0.99,
                sparse_optimization=sparse_optimization
            )

            self.p_time = self.generate_P(
                self.move,
                self.minigrid_env.custom_grid,
                self.allowed_actions,
                benchmark=benchmark_p,
                num_threads=threads
            )
            self._generate_R()
            print(f"Created LMDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
                    
        else:
            super().__init__(
                num_states=lmdp.num_states,
                num_terminal_states=lmdp.num_terminal_states,
                s0=lmdp.s0,
                lmbda=lmdp.lmbda,
                sparse_optimization=False # TODO: update
            )
            self.P = lmdp.P
            self.R = lmdp.R
    
    
    # TODO: generalize, as this is shared among all MDP, LMDP and LMDP_TDR
    def remove_unreachable_states(self):
        print("Going to remove unreachable states")
        
        reachable_states = set()
        queue = list(self.minigrid_env.custom_grid.terminal_states)

        for terminal_state in queue:
            reachable_states.add(terminal_state)

        while queue:
            current_state = queue.pop(0)
            for action in self.allowed_actions:
                next_state, _, _ = self.move(current_state, action)
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    queue.append(next_state)

        states = [state for state in self.minigrid_env.custom_grid.states if state in reachable_states]
        terminal_states = [state for state in self.minigrid_env.custom_grid.terminal_states if state in reachable_states]

        removed_states = len(self.minigrid_env.custom_grid.states) - len(states)
        print(f"Removing {removed_states} states")

        self.minigrid_env.custom_grid.states = states
        self.minigrid_env.custom_grid.terminal_states = terminal_states
        self.minigrid_env.custom_grid.generate_state_index_mapper()

    
    # TODO: same as MinigridMDP
    def move(self, state: State, action: int):
        orientation = state.properties["orientation"]
        y, x, curr_layout = state.y, state.x, state.layout
        
        next_state = State(y, x, layout=deepcopy(state.layout), **state.properties)
        
        if action in [MinigridActions.ROTATE_LEFT, MinigridActions.ROTATE_RIGHT]:
            next_state.properties["orientation"] = (orientation + (1 if action == MinigridActions.ROTATE_RIGHT else -1)) % 4
            
            return next_state, True, False
        
        dy, dx = self.OFFSETS[orientation]
        new_y, new_x = y + dy, x + dx
        next_object = curr_layout.get((new_x, new_y)) if curr_layout else None
        
        if action == MinigridActions.FORWARD:
            if next_object is not None and next_object.type == "key":
                # If there is a key in the next state, the agent remains at the same state.
                in_bounds = True
            
            elif next_object is not None and next_object.type == "door" and not state.properties[f"{next_object.color}_door_{next_object.id}"]:
                # If in the next state there is a door and it is not opened, the agent remains where it is
                in_bounds = True
            
            else:
                # The agent moves as usually
                next_state.y = y + dy
                next_state.x = x + dx
                in_bounds = self.minigrid_env.custom_grid.is_valid(next_state)
                if not in_bounds: next_state = state
        
            return next_state, in_bounds, self.minigrid_env.custom_grid.is_terminal(next_state)

        elif action == MinigridActions.PICKUP:
            # If the agent is facing a key, it gets it. Otherwise, it remains at the same state
            if not curr_layout: return next_state, True, False
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.minigrid_env.custom_grid.get_num_keys() - 1
            if next_object is not None and next_object.type == "key" and not agent_has_key:
                # print("pickup")
                next_state.layout[(new_x, new_y)] = None
            
            return next_state, True, False
            
        elif action == MinigridActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the door is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            if not curr_layout: return next_state, True, False
            missing_key = [obj for obj in self.minigrid_env.custom_grid.objects if obj not in curr_layout.values()]
            if len(missing_key) == 0:
                return next_state, True, False
            
            missing_key: Object = missing_key[0]
            
            if next_object is not None and next_object.type == "door" and missing_key.color == next_object.color:
                # print("toggle")
                next_state.properties[f"{next_object.color}_door_{next_object.id}"] = not next_state.properties[f"{next_object.color}_door_{next_object.id}"]
            return next_state, True, False
        
        elif action == MinigridActions.DROP:
            # If the agent is wearing a key and the position towards which it is facing is an empty square, then the agent can drop the object
            if not curr_layout: return next_state, True, False
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.minigrid_env.custom_grid.get_num_keys() - 1
           
            if next_object is None and agent_has_key and self.minigrid_env.custom_grid.is_normal(State(new_y, new_x, curr_layout, **state.properties)):
                carrying_object = [obj for obj in self.minigrid_env.custom_grid.objects if obj.type == "key" and obj not in layout_keys][0]
            
                next_state.layout[(new_x, new_y)] = carrying_object
            
            return next_state, True, False
        else:
            # Drop and Done actions have no effect yet
            return next_state, True, False

                  
    def _generate_R(self):
        for state in range(self.num_non_terminal_states):
            state_repr = self.minigrid_env.custom_grid.states[state]
            if self.minigrid_env.custom_grid.is_cliff(state_repr):
                # For precision purposes, ensure that reward / self.lmbda is greater than np.log(np.finfo(np.float64).tiny) = -708
                self.R[state] = np.float64(-100)
            else:
                self.R[state] = np.float64(-10)



    def transition_action(self, state_idx, next_state_idx):
        curr_state = self.minigrid_env.custom_grid.state_index_mapper[state_idx]
        for action in self.allowed_actions:
            move_state, _, _ = self.move(curr_state, action)
            next_state = self.minigrid_env.custom_grid.state_index_mapper[next_state_idx]
            if type(next_state) == State:
                if move_state == next_state:
                    return action
            else:
                if move_state.y == next_state[0] and move_state.x == next_state[1]:
                    return action
                
        return 0

    
    def states_to_goal(self) -> list[int]:
        """
        Returns the indices of the states that lead to the solution based on the derived policy
        """
        curr_state = self.minigrid_env.custom_grid.state_index_mapper[self.s0]
        curr_state_idx = self.s0
        states = [self.s0]
        
        while not self.minigrid_env.custom_grid.is_terminal(curr_state):        
            curr_state_idx = self.policy[curr_state_idx]
            curr_state = self.minigrid_env.custom_grid.state_index_mapper[curr_state_idx]
            states.append(curr_state_idx)
        
        return states
        
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None):
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)
        else:
            assert policies is not None
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)


class MinigridLMDP_TDR(LMDP_TDR):
    
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
        properties: dict[str, list] = {"orientation": [i for i in range(4)]},
        objects: list[Object] = None,
        sparse_optimization: bool = True,
        benchmark_p: bool = False,
        threads: int = 4
    ):
        
        self.num_actions = len(allowed_actions)
        self.allowed_actions = allowed_actions
        
        self.minigrid_env = CustomMinigridEnv(grid_size=grid_size, render_mode="rgb_array", map=map, properties=properties, objects=objects)
        self.remove_unreachable_states()
        
        
        
        self.num_states = self.minigrid_env.custom_grid.get_num_states()
        
        start_pos = self.minigrid_env.custom_grid.start_pos
        
        
        super().__init__(
            self.num_states,
            num_terminal_states=self.minigrid_env.custom_grid.get_num_terminal_states(),
            s0=0,
            # lmbda=0.99,
            sparse_optimization=sparse_optimization
        )

        self.p_time = self.generate_P(
            self.minigrid_env.custom_grid.states,
            self.move,
            self.minigrid_env.custom_grid,
            self.allowed_actions,
            benchmark=benchmark_p,
            num_threads=threads
        )
        self._generate_R()
        print(f"Created LMDP with {self.num_states} states. ({self.num_terminal_states} terminal and {self.num_non_terminal_states} non-terminal)")
    
    
    # TODO: generalize, as this is shared among all MDP, LMDP and LMDP_TDR
    def remove_unreachable_states(self):
        print("Going to remove unreachable states")
        reachable_states = set()
        queue = []
        for terminal_state in self.minigrid_env.custom_grid.terminal_states:
            reachable_states.add(terminal_state)
            queue.append(terminal_state)
        
        while queue:
            current_state = queue.pop(0)
            for action in self.allowed_actions:
                next_state, _, _ = self.move(current_state, action)
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    queue.append(next_state)
        
        states = list(reachable_states - set(self.minigrid_env.custom_grid.terminal_states))
        print(f"Removing {len(self.minigrid_env.custom_grid.states) - len(states)} states")
        self.minigrid_env.custom_grid.states = states
        self.minigrid_env.custom_grid.generate_state_index_mapper()   
    
    # TODO: same as MinigridMDP
    def move(self, state: State, action: int):
        orientation = state.properties["orientation"]
        y, x, curr_layout = state.y, state.x, state.layout
        
        next_state = State(y, x, layout=deepcopy(state.layout), **state.properties)
        
        if action in [MinigridActions.ROTATE_LEFT, MinigridActions.ROTATE_RIGHT]:
            next_state.properties["orientation"] = (orientation + (1 if action == MinigridActions.ROTATE_RIGHT else -1)) % 4
            
            return next_state, True, False
        
        dy, dx = self.OFFSETS[orientation]
        new_y, new_x = y + dy, x + dx
        next_object = curr_layout.get((new_x, new_y)) if curr_layout else None
        
        if action == MinigridActions.FORWARD:
            if next_object is not None and next_object.type == "key":
                # If there is a key in the next state, the agent remains at the same state.
                in_bounds = True
            
            elif next_object is not None and next_object.type == "door" and not state.properties[f"{next_object.color}_door_{next_object.id}"]:
                # If in the next state there is a door and it is not opened, the agent remains where it is
                in_bounds = True
            
            else:
                # The agent moves as usually
                next_state.y = y + dy
                next_state.x = x + dx
                in_bounds = self.minigrid_env.custom_grid.is_valid(next_state)
                if not in_bounds: next_state = state
        
            return next_state, in_bounds, self.minigrid_env.custom_grid.is_terminal(next_state)

        elif action == MinigridActions.PICKUP:
            # If the agent is facing a key, it gets it. Otherwise, it remains at the same state
            if not curr_layout: return next_state, True, False
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.minigrid_env.custom_grid.get_num_keys() - 1
            if next_object is not None and next_object.type == "key" and not agent_has_key:
                # print("pickup")
                next_state.layout[(new_x, new_y)] = None
            
            return next_state, True, False
            
        elif action == MinigridActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the door is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            if not curr_layout: return next_state, True, False
            missing_key = [obj for obj in self.minigrid_env.custom_grid.objects if obj not in curr_layout.values()]
            if len(missing_key) == 0:
                return next_state, True, False
            
            missing_key: Object = missing_key[0]
            
            if next_object is not None and next_object.type == "door" and missing_key.color == next_object.color:
                # print("toggle")
                next_state.properties[f"{next_object.color}_door_{next_object.id}"] = not next_state.properties[f"{next_object.color}_door_{next_object.id}"]
            return next_state, True, False
        
        elif action == MinigridActions.DROP:
            # If the agent is wearing a key and the position towards which it is facing is an empty square, then the agent can drop the object
            if not curr_layout: return next_state, True, False
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.minigrid_env.custom_grid.get_num_keys() - 1
           
            if next_object is None and agent_has_key and self.minigrid_env.custom_grid.is_normal(State(new_y, new_x, curr_layout, **state.properties)):
                carrying_object = [obj for obj in self.minigrid_env.custom_grid.objects if obj.type == "key" and obj not in layout_keys][0]
            
                next_state.layout[(new_x, new_y)] = carrying_object
            
            return next_state, True, False
        else:
            # Drop and Done actions have no effect yet
            return next_state, True, False

                  
    def _generate_R(self):
        for state in range(self.num_non_terminal_states):
            for action in range(self.num_actions):
                state_repr = self.minigrid_env.custom_grid.states[state]
                next_state, _, terminal = self.move(state_repr, action)
                if terminal:
                    next_state_idx = self.minigrid_env.custom_grid.terminal_state_idx(next_state)
                else:
                    next_state_idx = self.minigrid_env.custom_grid.states.index(next_state)
                
                if self.minigrid_env.custom_grid.is_cliff(state_repr):
                    self.R[state, next_state_idx] = -10
                else:
                    
                    self.R[state, next_state_idx] = -1

        # Matrix R is now sparese as well, so if sparse_optimization is activated, we convert it.
        if self.sparse_optimization:
            print("Converting R into sparse matrix...")
            print(f"Memory usage before conversion: {getsizeof(self.R):,} bytes")
            self.R = csr_matrix(self.R)
            print(f"Memory usage after conversion: {getsizeof(self.R):,} bytes")


    def transition_action(self, state_idx, next_state_idx):
        curr_state = self.minigrid_env.custom_grid.state_index_mapper[state_idx]
        for action in self.allowed_actions:
            move_state, _, _ = self.move(curr_state, action)
            next_state = self.minigrid_env.custom_grid.state_index_mapper[next_state_idx]
            if type(next_state) == State:
                if move_state == next_state:
                    return action
            else:
                if move_state.y == next_state[0] and move_state.x == next_state[1]:
                    return action
                
        return 0
    
    
    def visualize_policy(self, policies: list[tuple[int, np.ndarray]] = None, num_times: int = 10, save_gif: bool = False, save_path: str = None):
        assert not save_gif or save_path is not None, "Must specify save path"
        if not hasattr(self, "V") and policies is None:
            print(f"Computing value function...")
            self.compute_value_function()
            self.minigrid_env.visualize_policy(policies=[[0, self.policy]], num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)
        else:
            assert policies is not None
            self.minigrid_env.visualize_policy(policies=policies, num_times=num_times, save_gif=save_gif, save_path=save_path, model=self)

    