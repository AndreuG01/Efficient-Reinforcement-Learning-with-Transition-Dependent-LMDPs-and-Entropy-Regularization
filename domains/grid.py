from utils.state import State, Object
from itertools import product, combinations, permutations
import copy

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
    - POSITIONS_CHAR (dict): Maps characters to `CellType` values.
    - map (list[str]): The textual representation of the grid.
    - char_positions (dict): Maps `CellType` values to their corresponding characters.
    - positions (dict): Stores grid positions grouped by `CellType`.
    - start_pos (tuple[int, int]): The starting position in the grid.
    - goal_pos (list[tuple[int, int]]): List of goal positions in the grid.
    - size_x (int): The width of the grid.
    - size_y (int): The height of the grid.
    - state_index_mapper (dict): Maps state indices to grid positions.
    """
    
    POSITIONS_CHAR = {
        " ": CellType.NORMAL,
        "#": CellType.WALL,
        "S": CellType.START,
        "G": CellType.GOAL,
        "C": CellType.CLIFF
    }

    def __init__(self, map: list[str] = None, objects = list[Object], grid_size: int = 3, properties: dict[str, list] = None):
        """
        Initializes the grid with either a predefined map or generates a simple grid.

        Args:
        - map (list[str], optional): A list of strings representing the grid.
        - grid_size (int, optional): The size of the grid if generating a simple grid (default is 3).
        """
        self.map = map
        self.char_positions = {v: k for k, v in self.POSITIONS_CHAR.items()}
        
        self.positions = {k: [] for k in self.POSITIONS_CHAR.values()}
        self.state_properties = {} if properties is None else properties
        self.objects = objects if objects is not None else []
        
        for id, object in enumerate(self.objects):
            object.id = id
        
            
        self.__extend_properties()
        
        if self.map is None:
            self.generate_simple_grid(grid_size=grid_size)
        else:
            self.load_from_map()
            

        self.layout_combinations = self._get_layout_combinations()

        
        
        self.layout_combinations = self._get_layout_combinations()
        self.positions[CellType.NORMAL] = self._expand_state(CellType.NORMAL)
        # exit()
        self.positions[CellType.WALL] = self._expand_state(CellType.WALL)
        self.positions[CellType.START] = self._expand_state(CellType.START)
        self.positions[CellType.GOAL] = self._expand_state(CellType.GOAL)
        
    
        
        # self.positions[CellType.WALL] = [State(state.y, state.x, layout=layout, **dict(zip(list(self.state_properties.keys()), values))) for state in self.positions[CellType.WALL] for values in self._get_property_combinations() for layout in self.layout_combinations]
        # self.positions[CellType.START] = [State(state.y, state.x, layout=layout, **dict(zip(list(self.state_properties.keys()), values))) for state in self.positions[CellType.START] for values in self._get_property_combinations() for layout in self.layout_combinations]
        # self.positions[CellType.GOAL] = [State(state.y, state.x, layout=layout, **dict(zip(list(self.state_properties.keys()), values))) for state in self.positions[CellType.GOAL] for values in self._get_property_combinations() for layout in self.layout_combinations]
        # for layout, property in zip(self.properties_combinations, self.layout_combinations):
            
        print(self.get_num_states())
        # for idx, state in enumerate(self.positions[CellType.NORMAL]):
        #     state.layout = self.layout_combinations[idx % len(self.layout_combinations)]
        
        
        
        self.generate_state_index_mapper()
    
    
    def _expand_state(self, type):
        tmp = []
        for state in self.positions[type]:
            for values, layout in product(self._get_property_combinations(), self.layout_combinations):
                # if not self.__validate_comb(layout, dict(zip(list(self.state_properties.keys()), values))): continue
                # print(f"x={state.x}, y={state.y}, {layout}, properties={dict(zip(list(self.state_properties.keys()), values))}")
                tmp.append(State(state.y, state.x, layout=layout, **dict(zip(list(self.state_properties.keys()), values))))
        return tmp
        
        
    def __validate_comb(self, layout, properties):
        
        for k, object in layout.items():
            if object is not None and object.type == "key" and str(object) in properties.keys():
                
                return not properties[str(object)]
    
        return True
    

    def _get_layout_combinations(self):
        if len(self.objects) == 0: return [None]
        key_objects = [copy.deepcopy(obj) for obj in self.objects if obj.type == "key"]
        door_objects = [obj for obj in self.objects if obj.type == "door"]
        valid_positions = list(set([(state.x, state.y) for state in self.positions[CellType.NORMAL] if state not in self.positions[CellType.CLIFF] and state not in self.positions[CellType.START]]))

        # Get door positions and remove them from valid positions
        door_positions = {(obj.y, obj.x) for obj in door_objects}
        valid_positions = [pos for pos in valid_positions if pos not in door_positions]

        # Initialize first placement with the original positions of objects
        first_placement = {pos: None for pos in valid_positions}
        
        for obj in door_objects:
            first_placement[(obj.y, obj.x)] = obj  # Keep doors fixed
        
        for key_obj in key_objects:
            first_placement[(key_obj.y, key_obj.x)] = key_obj  # Place keys at their initial positions

        all_placements = [first_placement]  # First placement is the initial state

        # Generate all possible ways to place all keys (n keys)
        for key_positions in permutations(valid_positions, len(key_objects)):
            placement = {pos: None for pos in valid_positions}
            
            # Place doors
            for obj in door_objects:
                placement[(obj.y, obj.x)] = obj
            
            # Place keys
            for key_obj, pos in zip(key_objects, key_positions):
                # key_obj = copy.deepcopy(key_obj)
                # key_obj.y, key_obj.x = pos  # Swap coordinates
                placement[pos] = key_obj
            
            all_placements.append(placement)

        # Generate all possible ways to place n-1 keys
        for r in [len(key_objects) - 1]:  # Only need n-1 cases
            for key_subset in combinations(key_objects, r):
                for key_positions in permutations(valid_positions, r):
                    placement = {pos: None for pos in valid_positions}
                    
                    # Place doors
                    for obj in door_objects:
                        placement[(obj.y, obj.x)] = obj
                    
                    # Place n-1 keys
                    for key_obj, pos in zip(key_subset, key_positions):
                        # key_obj = copy.deepcopy(key_obj)
                        # key_obj.y, key_obj.x = pos  # Swap coordinates
                        placement[pos] = key_obj
                    
                    all_placements.append(placement)

        return all_placements


    def __extend_properties(self):
        for object in self.objects:
            self.state_properties[f"{object.color}_{object.type}_{object.id}"] = [False, True]

    def _get_property_combinations(self):
        property_keys = list(self.state_properties.keys())
        property_values = [self.state_properties[key] for key in property_keys]
        combinations = product(*property_values)
        
        return combinations
    
    
    def _generate_states(self, x: int, y: int) -> list[State]:
        property_combinations = self._get_property_combinations()
        
        return [State(x, y, layout=None, properties=None)]
        # return [State(x, y, layout=None, **dict(zip(list(self.state_properties.keys()), values))) for values in property_combinations]

    def generate_simple_grid(self, grid_size: int):
        """
        Generates a simple grid with a border of walls, a start position (top left), and a single goal (bottom right).

        Args:
        - grid_size (int): The size of the inner grid (excluding walls).
        """
        
        self.start_pos = (1, 1)
        self.goal_pos = [(grid_size, grid_size)]
        self.size_x = grid_size + 2
        self.size_y = grid_size + 2
        
        for j in range(self.size_x):
            for i in range(self.size_y):
                if (i == 0 or i == self.size_y - 1) or (j == 0 or j == self.size_x - 1):
                    self.positions[CellType.WALL].extend(self._generate_states(i, j))
                elif (i, j) != self.goal_pos[0]:
                    self.positions[CellType.NORMAL].extend(self._generate_states(i, j))
                
        self.positions[CellType.START].append(State(self.start_pos[0], self.start_pos[1], layout=None, **{k: v[0] for k, v in self.state_properties.items()}))
        self.positions[CellType.GOAL] = self._generate_states(self.goal_pos[0][0], self.goal_pos[0][1])
        
        
    
    
    def load_from_map(self):
        """
        Loads a grid from a predefined map and initializes cell positions.

        Raises:
        - AssertionError: If the map rows are not of equal length.
        """
        assert all(len(row) == len(self.map[0]) for row in self.map), "Not all rows have the same length"
        for j, row in enumerate(self.map):
            for i, cell in enumerate(row):
                if cell == self.char_positions[CellType.START]: self.positions[CellType.NORMAL].extend(self._generate_states(i, j))
                if cell == self.char_positions[CellType.CLIFF]: self.positions[CellType.NORMAL].extend(self._generate_states(i, j))
                # if cell == self.char_positions[CellType.KEY]: self.positions[CellType.NORMAL].extend(self._generate_states(i, j))
                # if cell == self.char_positions[CellType.DOOR]: self.positions[CellType.NORMAL].extend(self._generate_states(i, j))
                self.positions[self.POSITIONS_CHAR[cell]].extend(self._generate_states(i, j))
        
        start_state = self.positions[CellType.START][0]
        goal_states = self.positions[CellType.GOAL]
        self.start_pos: tuple[int, int] = (start_state.y, start_state.x)
        self.goal_pos = [(goal_state.x, goal_state.y) for goal_state in goal_states]
        self.size_x = len(self.map)
        self.size_y = len(self.map[0])
    
    
    def get_active_state_object(self, x: int, y: int) -> Object | None:
        
        for k, v in self.positions.items():
            for idx, state in enumerate(v):
                if state.x == x and state.y == y and state.object is not None and state.object.active:
                    return self.positions[k][idx].object
        # for pos in list(self.positions.values()):
        #     for state in pos:
        #         if state.x == x and state.y == y:
        #             if state.object is not None and state.object.active:
        #                 return state.object
        return None
    
    def set_state_object_visibility(self, x: int, y: int, object: Object, visibility) -> None:
        states = self.__get_states_at_pos(x, y)
        # for k, v in (self.positions.items()):
        for idx, state in enumerate(states):
            if state.x == x and state.y == y:
                if state.object is not None and state.object == object:
                    state.object.active = visibility
                    # self.positions[k][idx].object.active = not object.active

    
    def __get_states_at_pos(self, x: int, y: int) -> list[State]:
        target_states = []
        for state_type, states in self.positions.items():
            for idx, state in enumerate(states):
                if state.x == x and state.y == y:
                    target_states.append(state)
        return target_states
    
    def get_num_states(self) -> int:
        """
        Returns the total number of navigable states (normal cells + goal cells).

        Returns:
        - int: Total number of states.
        """
        return len(self.positions[CellType.NORMAL]) + len(self.positions[CellType.GOAL])
    
    
    def get_num_terminal_states(self):
        """
        Returns the number of terminal states (goal cells).

        Returns:
        - int: Number of terminal states.
        """
        return len(self.positions[CellType.GOAL])
    
    
    def generate_state_index_mapper(self) -> dict[int, tuple[int, int]]:
        """
        Generates a mapping of state indices to grid positions.

        Returns:
        - dict[int, tuple[int, int]]: The state index-to-position mapping.
        """
        self.state_index_mapper = {} # Maps the state index to the position in the grid.
        for count, pos in enumerate(self.positions[CellType.NORMAL] + self.positions[CellType.GOAL]):
            self.state_index_mapper[count] = pos
    
    
    def state_has_key(self, state: State) -> bool:
        # Check if a state has any key picked
        keys_picked = []
        for k, v in state.properties.items():
            if "key" in k:
                keys_picked.append(v)
        return any(keys_picked)
    
        
        
    
    
    def state_has_key_color(self, state: State, color: str) -> bool:
        keys_picked = []
        for k, v in state.properties.items():
            if f"{color}_key" in k:
                keys_picked.append(v)
        return any(keys_picked)
    
    
    def remove_object(self, x: int, y: int) -> Object:
        return_obj = None
        for idx, state in enumerate(self.positions[CellType.NORMAL]):
            if state.x == x and state.y == y:
                assert state.object is not None
                self.positions[CellType.NORMAL][idx].object = None
                return_obj = state.object
        return return_obj
            
    def add_object(self, x: int, y: int, object: Object) -> None:
        for idx, state in enumerate(self.positions[CellType.NORMAL]):
            if state.x == y and state.y == x:
                self.positions[CellType.NORMAL][idx].object = object
                
            
    def get_carrying_object(self, state: State) -> set[Object]:
        
        carrying_elems = [k for k, v in state.properties.items() if "key" in k and v ]
        if len(carrying_elems) == 0:
            return None
        
        carrying_elems = carrying_elems[0].split("_")
            
        for curr_obj in self.objects:
            if curr_obj.color == carrying_elems[0] and curr_obj.type == carrying_elems[1] and curr_obj.id == int(carrying_elems[2]):
                return curr_obj
        
        
    
    def is_valid(self, state: State) -> bool:
        """
        Checks if a position is valid (i.e., not a wall).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is valid, False otherwise.
        """
        return state not in self.positions[CellType.WALL]


    def is_terminal(self, state: State) -> bool:
        """
        Checks if a position is a terminal state (i.e., a goal position).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is terminal, False otherwise.
        """
        return state in self.goal_pos
    
    def is_key(self, state: State) -> bool:
        """
        Checks if a position is a key

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is a key, False otherwise.
        """
        return state.object is not None and state.object.type == "key"
    
    
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
        return state in self.positions[CellType.NORMAL] and state not in self.positions[CellType.CLIFF]
    
    
    def print_grid(self):
        """
        Prints a textual representation of the grid world, showing the types of cells (normal, wall, start, goal).
        """
        for j in range(self.size_x):
            for i in range(self.size_y):
                type = [k for k, v in self.grid.positions.items() if (i, j) in v][-1]
                print(self.char_positions[type], end="")
            print()
    