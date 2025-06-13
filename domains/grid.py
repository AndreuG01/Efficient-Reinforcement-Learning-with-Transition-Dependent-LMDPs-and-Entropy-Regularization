from utils.state import State, Object
from itertools import product, combinations, permutations
from copy import deepcopy
import copy
from typing import Literal
from utils.maps import Map
import numpy as np
from tqdm import tqdm
import time
from utils.coloring import TerminalColor
from collections import deque

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

class GridWorldActions:
    """
    Possible actions:
        Action number   |   Action type | Action description     |   Keyboard key     
        ------------------------------------------------------------------------
              0         |   up          |   move up              |   Up / w
              1         |   right       |   move right           |   Right / d
              2         |   left        |   move left            |   Left / a
              3         |   down        |   move down            |   Down / s
              4         |   pickup      |   Pickup an object     |   Tab
              5         |   drop        |   Drop an object       |   Left shift
              6         |   toggle      |   Toggle               |   Space
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    PICKUP = 4
    DROP = 5
    TOGGLE = 6
    
    @classmethod
    def get_actions(cls) -> list[int]:
        return [value for key, value in cls.__dict__.items() if type(value) == int]

class CellType:
    """
    Represents different types of cells in the grid in an Enum like style.

    Attributes:
    - NORMAL (int): Represents a normal navigable cell.
    - WALL (int): Represents a wall or an obstacle.
    - START (int): Represents the starting position.
    - GOAL (int): Represents a goal position.
    - CLIFF (int): Represents a dangerous or non-navigable cliff cell.
    
    Note:
    Additional cell types can be added in the future as needed.
    """
    NORMAL = 0
    WALL = 1
    START = 2
    GOAL = 3
    CLIFF = 4
    # ... --> can be extended in the future to acount for more wall types

class CustomGrid:
    """
    Represents a customizable grid world for navigation and simulation.

    Attributes:
        POSITIONS_CHAR (dict): Maps characters to `CellType` values.
        type (str): The type of grid, either "gridworld" or "minigrid".
        map (Map): The map object containing the grid layout and objects.
        allowed_actions (list[int]): List of allowed actions for the agent.
        verbose (bool): If True, prints progress information during state generation.
        positions (dict[CellType, list[tuple[int, int]]]): Dictionary mapping cell types to their positions in the grid.
        objects (list[Object]): List of objects present in the grid.
        state_properties (dict[str, list]): Properties of the states.
        states (list[State]): List of all navigable states in the grid.
        terminal_states (list[State]): List of terminal states (goal positions).
        original_states (list[State]): Copy of the original states before any modifications.
        original_terminal_states (list[State]): Copy of the original terminal states before any modifications.
        max_state_space_size (int): Maximum size of the state space.
        state_index_mapper (dict[int, State]): Maps state indices to their corresponding `State` objects.
        start_pos (tuple[int, int]): Starting position of the agent in the grid.
        goal_pos (list[tuple[int, int]]): List of goal positions in the grid.        
    """
    
    POSITIONS_CHAR = {
        " ": CellType.NORMAL,
        "#": CellType.WALL,
        "S": CellType.START,
        "G": CellType.GOAL,
        "C": CellType.CLIFF
    }

    def __init__(
        self,
        type: Literal["gridworld", "minigrid"],
        map: Map,
        properties: dict[str, list] = {},
        allowed_actions: list[int] = None,
        verbose: bool = True
    ):
        """
        Initializes the grid with either a predefined map or generates a simple grid.

        Args:
            type (str): The type of grid, either "gridworld" or "minigrid".
            map (Map): The map object containing the grid layout and objects.
            properties (dict[str, list]): Properties of the states.
            allowed_actions (list[int]): List of allowed actions for the agent.
            verbose (bool): If True, prints progress information during state generation.
        """
        self.verbose = verbose
        self.type = type
        self.map = map
        self.char_positions = {v: k for k, v in self.POSITIONS_CHAR.items()}
        
        self.positions = {k: [] for k in self.POSITIONS_CHAR.values()}
        self.objects = self.map.objects
        self.allowed_actions = allowed_actions
        
        for id, object in enumerate(self.objects):
            object.id = id
        
        self.state_properties = self.__extend_properties(properties)
        
        
        self.load_from_map()
            

        self.layout_combinations = self._get_layout_combinations()
        
        
        self.states, self.terminal_states = self._generate_states()
        
        self.original_states = self.states.copy()
        self.original_terminal_states = self.terminal_states.copy()
        self.max_state_space_size = len(self.states) + len(self.terminal_states)
        
        self.remove_unreachable_states()
        
    
    def _generate_states(self) -> tuple[list[State], list[State]]:
        """
        Generates all possible states in the grid based.
        
        Returns:
            tuple[list[State], list[State]]: A tuple containing two lists:
                - The first list contains all normal states.
                - The second list contains all terminal states (goal states).
        """
        states = []
        terminal_states = []
        for pos in tqdm(self.positions[CellType.NORMAL], desc="Generating normal states", total=len(self.positions[CellType.NORMAL]), disable=not self.verbose):
            for values, layout in product(self._get_property_combinations(), self.layout_combinations):
                curr_dict = dict(zip(list(self.state_properties.keys()), values))
                curr_state = State(pos[1], pos[0], layout=layout, **curr_dict)
                states.append(curr_state)
        
        for pos in tqdm(self.positions[CellType.GOAL], desc="Generating goal states", total=len(self.positions[CellType.GOAL]), disable=not self.verbose):
            for values, layout in product(self._get_property_combinations(), self.layout_combinations):
                terminal_states.append(State(pos[1], pos[0], layout=layout, **dict(zip(list(self.state_properties.keys()), values))))
                
        return states, terminal_states        


    def _get_layout_combinations(self) -> list[dict[tuple[int, int], Object]]:
        """
        Generates all possible layout combinations for the grid based on the positions of keys and doors.
        
        Returns:
            list[dict[tuple[int, int], Object]]: A list of dictionaries representing all possible layout combinations.
        """
        
        if len(self.objects) == 0: return [None]
        key_objects = [copy.deepcopy(obj) for obj in self.objects if obj.type == "key"]
        door_objects = [obj for obj in self.objects if obj.type == "door"]
        valid_positions = list(set([(pos[1], pos[0]) for pos in self.positions[CellType.NORMAL] if pos not in self.positions[CellType.CLIFF]]))
        
        door_positions = {(obj.y, obj.x) for obj in door_objects}
        valid_positions = [pos for pos in valid_positions if pos not in door_positions]

        # Initialize first placement with the original positions of objects
        first_placement = {pos: None for pos in valid_positions}
        for obj in door_objects:
            first_placement[(obj.y, obj.x)] = obj
        
        for key_obj in key_objects:
            first_placement[(key_obj.y, key_obj.x)] = key_obj

        all_placements = [first_placement]  # First placement is the initial state

        # Generate all possible ways to place all n keys
        for key_positions in permutations(valid_positions, len(key_objects)):
            placement = {pos: None for pos in valid_positions}
            # Doors
            for obj in door_objects:
                placement[(obj.y, obj.x)] = obj
            
            # Keys
            for key_obj, pos in zip(key_objects, key_positions):    
                placement[pos] = key_obj
            
            if placement not in all_placements:
                # Done to avoid adding the initial configuration once again
                all_placements.append(placement)

        # Generate all possible ways to place n-1 keys
        for key_subset in combinations(key_objects, max(len(key_objects) - 1, 0)):
            for key_positions in permutations(valid_positions, max(len(key_objects) - 1, 0)):
                placement = {pos: None for pos in valid_positions}
                
                # Doors
                for obj in door_objects:
                    placement[(obj.y, obj.x)] = obj
                
                # Keys
                for key_obj, pos in zip(key_subset, key_positions):
                    placement[pos] = key_obj
                
                all_placements.append(placement)

        return all_placements


    def __extend_properties(self, properties: dict[str, list]) -> dict[str, list]:
        """
        Extends the properties dictionary with additional properties based on the objects in the grid.
        It sets the door properties for each object, indicating whether the door is open or closed.
        
        Args:
            properties (dict[str, list]): The initial properties dictionary.
        
        Returns:
            dict[str, list]: The extended properties dictionary with additional properties for each object.
        """
        # Do not consider key objects
        for object in self.objects:
            if object.type == "key": continue
            properties[f"{object.color}_{object.type}_{object.id}"] = [False, True]
        
        return properties

    def _get_property_combinations(self) -> list[tuple]:
        """
        Generates all combinations of the state properties defined in the grid.
        
        Returns:
            list[tuple]: A list of tuples, where each tuple contains a combination of state properties.
        """
        property_keys = list(self.state_properties.keys())
        property_values = [self.state_properties[key] for key in property_keys]
        combinations = product(*property_values)
        
        return combinations
        
    
    def load_from_map(self):
        """
        Loads a grid from a predefined map and initializes cell positions.

        Raises:
        - AssertionError: If the map rows are not of equal length.
        """
        layout = self.map.layout
        assert all(len(row) == len(layout[0]) for row in layout), "Not all rows have the same length"
        for j, row in enumerate(layout):
            for i, cell in enumerate(row):
                if cell == self.char_positions[CellType.START]: self.positions[CellType.NORMAL].append((i, j))
                if cell == self.char_positions[CellType.CLIFF]: self.positions[CellType.NORMAL].append((i, j))
                self.positions[self.POSITIONS_CHAR[cell]].append((i, j))
        
        
        goal_states = self.positions[CellType.GOAL]
        self.start_pos: tuple[int, int] = self.positions[CellType.START][0]
        self.goal_pos = [(goal_state[0], goal_state[1]) for goal_state in goal_states]
        self.size_x = len(layout)
        self.size_y = len(layout[0])
    
    
    def move(self, state: State, action: int):
        """
        Computes the next position after performing an action in a given state.
        It acts as a wrapper method that calls the appropriate move method based on the grid type.
        
        Args:
            state (State): The current state of the agent.
            action (int): The action to be performed.
        
        Returns:
            tuple[State, bool, bool]: A tuple containing:
                - The next state after the action.
                - A boolean indicating whether the move is valid.
                - A boolean indicating whether the agent has reached a terminal state.
        
        """
        if self.type == "gridworld":
            return self.__move_gridworld(state, action)
        else:
            return self.__move_minigrid(state, action)
    
    
    def __move_gridworld(
        self,
        state: State,
        action: int,
        offsets: dict ={0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ) -> tuple[State, bool, bool]:
        """
        Computes the next position after performing an action in a GridWorld environment.

        Args:
            state (State): The current state of the agent.
            action (int): The action to be performed.
            offsets (dict): A dictionary mapping actions to their corresponding (dy, dx) offsets.

        Returns:
            tuple[State, bool, bool]: A tuple containing:
                - The next state after the action.
                - A boolean indicating whether the move is valid.
                - A boolean indicating whether the agent has reached a terminal state.
        """
        y, x, curr_layout = state.y, state.x, state.layout
        next_state = State(y, x, layout=deepcopy(state.layout), **state.properties)
        
        if action == GridWorldActions.UP or action == GridWorldActions.RIGHT or action == GridWorldActions.DOWN or action == GridWorldActions.LEFT:
            dy, dx = offsets[action]
            next_object = curr_layout.get((next_state.y + dy, next_state.x + dx)) if curr_layout else None
            if next_object is not None and next_object.type == "key":
                # If there is a key in the next state, the agent remains at the same state.
                in_bounds = True
            
            elif next_object is not None and next_object.type == "door" and not state.properties[f"{next_object.color}_door_{next_object.id}"]:
                # If in the next state there is a door and it is not opened, the agent remains where it is
                in_bounds = True
            else:
                next_state.y = next_state.y + dy
                next_state.x = next_state.x + dx
                
                in_bounds = self.is_valid(next_state)
                if not in_bounds: next_state = state

            return next_state, in_bounds, self.is_terminal(next_state)
        
        elif action == GridWorldActions.PICKUP:
            # If the agent is facing a key, it gets it. Otherwise, it remains at the same state
            if not curr_layout: return next_state, True, self.is_terminal(next_state)
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.get_num_keys() - 1
            
            for dy, dx in offsets.values():
                new_x = next_state.x + dx
                new_y = next_state.y + dy
                next_object = curr_layout.get((new_y, new_x)) if curr_layout else None
                if next_object is not None and next_object.type == "key" and not agent_has_key:
                    # An object has been found
                    next_state.layout[(new_y, new_x)] = None
                    break
            
            return next_state, True, self.is_terminal(next_state)
                
        elif action == GridWorldActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the door is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            if not curr_layout: return next_state, True, self.is_terminal(next_state)
            missing_key = [obj for obj in self.objects if obj not in curr_layout.values()]
            if len(missing_key) == 0:
                return next_state, True, self.is_terminal(next_state)
            
            missing_key: Object = missing_key[0]
            for dy, dx in offsets.values():
                new_x, new_y = next_state.x + dx, next_state.y + dy
                next_object = curr_layout.get((new_y, new_x)) if curr_layout else None
                if next_object is not None and next_object.type == "door" and missing_key.color == next_object.color:

                    next_state.properties[f"{next_object.color}_door_{next_object.id}"] = not next_state.properties[f"{next_object.color}_door_{next_object.id}"]
            
            return next_state, True, self.is_terminal(next_state)
        
        elif action == GridWorldActions.DROP:
            # If the agent is wearing a key and the position towards which it is facing is an empty square, then the agent can drop the object
            if not curr_layout: return next_state, True, self.is_terminal(next_state)
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.get_num_keys() - 1

            for dy, dx in offsets.values():
                new_x, new_y = next_state.x + dx, next_state.y + dy
                next_object = curr_layout.get((new_y, new_x)) if curr_layout else None
            
                if next_object is None and agent_has_key and self.is_normal(State(new_y, new_x, curr_layout, **state.properties)):
                    carrying_object = [obj for obj in self.objects if obj.type == "key" and obj not in layout_keys][0]
                
                    next_state.layout[(new_y, new_x)] = carrying_object
                    break
            
            return next_state, True, self.is_terminal(next_state)
    
        else:
            return next_state, True, self.is_terminal(next_state)
    
    def __move_minigrid(
        self,
        state: State,
        action: int,
        offsets: dict ={0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
    ) -> tuple[State, bool, bool]:
        """
        Computes the next position after performing an action in a MiniGrid environment.

        Args:
            state (State): The current state of the agent.
            action (int): The action to be performed.
            offsets (dict): A dictionary mapping actions to their corresponding (dy, dx) offsets.

        Returns:
            tuple[State, bool, bool]: A tuple containing:
                - The next state after the action.
                - A boolean indicating whether the move is valid.
                - A boolean indicating whether the agent has reached a terminal state.
        """
        orientation = state.properties["orientation"]
        y, x, curr_layout = state.y, state.x, state.layout
        
        next_state = State(y, x, layout=deepcopy(state.layout), **state.properties)
        
        if action in [MinigridActions.ROTATE_LEFT, MinigridActions.ROTATE_RIGHT]:
            next_state.properties["orientation"] = (orientation + (1 if action == MinigridActions.ROTATE_RIGHT else -1)) % 4
            
            return next_state, True, self.is_terminal(next_state)

        dy, dx = offsets[orientation]
        new_y, new_x = y + dy, x + dx
        next_object = curr_layout.get((new_y, new_x)) if curr_layout else None
        
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
                in_bounds = self.is_valid(next_state)
                if not in_bounds: next_state = state
        
            return next_state, in_bounds, self.is_terminal(next_state)

        elif action == MinigridActions.PICKUP:
            # If the agent is facing a key, it gets it. Otherwise, it remains at the same state
            if not curr_layout: return next_state, True, False
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.get_num_keys() - 1
            if next_object is not None and next_object.type == "key" and not agent_has_key:
                # print("pickup")
                next_state.layout[(new_y, new_x)] = None
            
            return next_state, self.is_valid(next_state), self.is_terminal(next_state)
            
        elif action == MinigridActions.TOGGLE:
            # If the agent is facing a door for which it has the key:
            #   - If the door is closed: it opens it.
            #   - If the door is opened: it closes it.
            # If the agent does not have the key, it remains where it is.
            if not curr_layout: return next_state, self.is_valid(next_state), self.is_terminal(next_state)
            missing_key = [obj for obj in self.objects if obj not in curr_layout.values()]
            if len(missing_key) == 0:
                return next_state, self.is_valid(next_state), self.is_terminal(next_state)
            
            missing_key: Object = missing_key[0]
            
            if next_object is not None and next_object.type == "door" and missing_key.color == next_object.color:
                # print("toggle")
                next_state.properties[f"{next_object.color}_door_{next_object.id}"] = not next_state.properties[f"{next_object.color}_door_{next_object.id}"]
            return next_state, self.is_valid(next_state), self.is_terminal(next_state)
        
        elif action == MinigridActions.DROP:
            # If the agent is wearing a key and the position towards which it is facing is an empty square, then the agent can drop the object
            if not curr_layout: return next_state, self.is_valid(next_state), self.is_terminal(next_state)
            layout_keys = [obj for obj in curr_layout.values() if type(obj) == Object and obj.type == "key"]
            agent_has_key = len(layout_keys) == self.get_num_keys() - 1
           
            if next_object is None and agent_has_key and self.is_normal(State(new_y, new_x, curr_layout, **state.properties)):
                carrying_object = [obj for obj in self.objects if obj.type == "key" and obj not in layout_keys][0]
            
                next_state.layout[(new_y, new_x)] = carrying_object
            
            return next_state, self.is_valid(next_state), self.is_terminal(next_state)
        else:
            # Done actions have no effect yet
            return next_state, self.is_valid(next_state), self.is_terminal(next_state)
    
    
    def _unrechable_info(self, steps: int, queue_len: int, reachable_len: int, last_count: int, last_time: float, start_time: float) -> tuple[int, float]:
        """
        Prints the information message shown during the removal of unreachable states
        
        Args:
            steps (int): The number of steps taken so far.
            queue_len (int): The length of the queue of states to be processed.
            reachable_len (int): The number of reachable states found so far.
            last_count (int): The last count of reachable states.
            last_time (float): The last time when the count was updated.
            start_time (float): The time when the process started.
        
        Returns:
            tuple[int, float]: A tuple containing the updated reachable length and the current time.
        """
        now = time.time()
        rate = (reachable_len - last_count) / (now - last_time) if now > last_time else 0
        elapsed = now - start_time

        msg = (
            f"Steps: {str(steps)}" +
            f" | In queue: {TerminalColor.colorize(str(queue_len), 'red')}" +
            f" | Reachable: {str(reachable_len)}" +
            f" | Discovery rate: {str(round(rate, 2))}/s" +
            f" | Elapsed: {str(round(elapsed, 1))}s"
        )
        self._print(msg.ljust(100), end="\r")
        return reachable_len, now

    def remove_unreachable_states(self) -> None:
        """
        Removes states that are unreachable from the start state.

        This function uses a breadth-first search (BFS) approach to find all reachable states and then removes the unreachable ones.

        Returns:
            None
        """
        start_state = [state for state in self.states if state.x == self.start_pos[0] and state.y == self.start_pos[1]][0]
        
        reachable_states = set()
        queue = [start_state]

        for terminal_state in queue:
            reachable_states.add(terminal_state)

        steps = 0
        start_time = time.time()
        last_count = 0
        last_time = start_time
        rate_interval = 500

        while queue:
            if steps % 100 == 0:
                if steps % rate_interval == 0:
                    last_count, last_time = self._unrechable_info(steps, len(queue), len(reachable_states), last_count, last_time, start_time)
                else:
                    self._unrechable_info(steps, len(queue), len(reachable_states), last_count, last_time, start_time)
            
            current_state = queue.pop(0)
            for action in self.allowed_actions:
                next_state, _, _ = self.move(current_state, action)
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    queue.append(next_state)
                steps += 1
        self._unrechable_info(steps, len(queue), len(reachable_states), last_count, last_time, start_time)
        self._print("")

        
        states = [state for state in self.states if state in reachable_states]
        terminal_states = [state for state in self.terminal_states if state in reachable_states]

        removed_states = len(self.states) + len(self.terminal_states) - len(reachable_states)
        self._print(f"Removing {removed_states} states")

        self.states = states
        self.terminal_states = terminal_states
        
        assert len(self.states) == len(set(self.states)), "There are duplicate non-terminal states"
        assert len(self.terminal_states) == len(set(self.terminal_states)), "There are duplicate terminal-states"
        
        self.generate_state_index_mapper()
    
    
    def transition_action(self, state_idx: int, next_state_idx: int, allowed_actions: list[int]) -> int:
        """
        Identifies the action that leads from one state index to another.

        Args:
            state_idx (int): Index of the starting state.
            next_state_idx (int): Index of the resulting state.
            allowed_actions (list[int]): List of allowed actions to consider.

        Returns:
            int: The action that causes the transition, or -1 if none match.
        """
        curr_state = self.state_index_mapper[state_idx]
        for action in allowed_actions:
            move_state, _, _ = self.move(curr_state, action)
            next_state = self.state_index_mapper[next_state_idx]
            if type(next_state) == State:
                if move_state == next_state:
                    return action
            else:
                if move_state.y == next_state[1] and move_state.x == next_state[0]:
                    return action
                
        return -1
        
    
    def shortest_path_length_to_goal(self, state: State) -> int:
        """
        Returns the length of the shortest path from the given state to any goal state.
        
        Args:
            state (State): The current state from which to calculate the shortest path length.
        
        Returns:
            int: The length of the shortest path to a goal state, or -1 if no path exists.
        """

        visited = set()
        queue = deque([(state, 0)])

        while queue:
            current_state, dist = queue.popleft()
            if self.is_terminal(current_state):
                return dist
            if current_state in visited:
                continue
            visited.add(current_state)
            for action in self.allowed_actions:
                next_state, valid, _ = self.move(current_state, action)
                if valid and next_state not in visited:
                    queue.append((next_state, dist + 1))
        return -1
    
    
    def get_num_states(self) -> int:
        """
        Returns the total number of navigable states (normal cells + goal cells).

        Returns:
        - int: Total number of states.
        """
        return len(self.states) + len(self.terminal_states)
    
    
    def get_num_terminal_states(self):
        """
        Returns the number of terminal states (goal cells).

        Returns:
        - int: Number of terminal states.
        """
        return len(self.terminal_states)
    
    
    def generate_state_index_mapper(self) -> dict[int, tuple[int, int]]:
        """
        Generates a mapping of state indices to grid positions.

        Returns:
        - dict[int, tuple[int, int]]: The state index-to-position mapping.
        """
       # TODO: maybe this is not useful anymore 
        self.state_index_mapper = {} # Maps the state index to the position in the grid.
        for count, pos in enumerate(self.states + self.terminal_states):
            self.state_index_mapper[count] = pos
    
    
    def get_state_pos(self, x: int, y: int) -> list[State]:
        return_states = []
        
        for state in self.states + self.terminal_states:
            if state.y == y and state.x == x:
                return_states.append(state)
        
        return return_states
        
    
    def is_valid(self, state: State) -> bool:
        """
        Checks if a position is valid (i.e., not a wall).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is valid, False otherwise.
        """
        return (state.x, state.y) not in self.positions[CellType.WALL]


    def is_terminal(self, state: State) -> bool:
        """
        Checks if a position is a terminal state (i.e., a goal position).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is terminal, False otherwise.
        """
        return state in self.terminal_states
    
    
    def is_key(self, state: State) -> bool:
        """
        Checks if a position is a key

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is a key, False otherwise.
        """
        for pos, obj in state.layout.items():
            if obj is not None and obj.type == "key" and (pos[0] == state.x and pos[1] == state.y):
                return True, obj
        return False, None
    
    
    def is_door(self, state: State) -> bool:
        """
        Checks if a position is a door.

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is a door, False otherwise.
        """
        return state.object is not None and state.object.type == "door"
    
    def is_normal(self, state: State) -> bool:
        return (state.x, state.y) in self.positions[CellType.NORMAL] and (state.x, state.y) not in self.positions[CellType.CLIFF]
    
    
    def is_cliff(self, state: State | tuple[int, int]) -> bool:
        if type(state) == State:
            return (state.x, state.y) in self.positions[CellType.CLIFF]
        else:
            return (state[1], state[0]) in self.positions[CellType.CLIFF]
    
    
    def get_num_keys(self) -> int:
        return len([obj for obj in self.objects if obj.type == "key"])
    
    def get_num_doors(self) -> int:
        return len([obj for obj in self.objects if obj.type == "door"])
    
    def print_grid(self):
        """
        Prints a textual representation of the grid world, showing the types of cells (normal, wall, start, goal).
        """
        for j in range(self.size_x):
            for i in range(self.size_y):
                type = [k for k, v in self.grid.positions.items() if (i, j) in v][-1]
                print(self.char_positions[type], end="")
            print()
    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)