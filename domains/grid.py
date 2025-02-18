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
        self.objects = objects if objects is not None else []
        
        for id, object in enumerate(self.objects):
            object.id = id
        
        self.state_properties = {} if properties is None else self.__extend_properties(properties)
        
        
        if self.map is None:
            self.generate_simple_grid(grid_size=grid_size)
        else:
            self.load_from_map()
            

        self.layout_combinations = self._get_layout_combinations()
        
        
        self.states, self.terminal_states = self._generate_states()
        
        self.generate_state_index_mapper()
    
    
    def _generate_states(self):
        states = []
        terminal_states = []
        for pos in self.positions[CellType.NORMAL]:
            for values, layout in product(self._get_property_combinations(), self.layout_combinations):
                # TODO: a state where there is a key at the same position as the agent is not valid

                # if obj is not None and obj.type == "key": continue
                
                curr_dict = dict(zip(list(self.state_properties.keys()), values))

                # if obj is not None and obj.type == "door" and not curr_dict[str(obj)]: continue
                # if (pos[1] == 1 and pos[0] == 1) and (layout[(2, 1)] is not None and layout[(3, 1)] is None and layout[(1, 1)] is None and layout[(4, 2)] is None and layout[(3, 3)] is None and layout[(3, 2)] is None and layout[(4, 1)] is None): continue
                # if not curr_dict["blue_door_0"] and (pos[1] == 3 and pos[0] == 3) and (layout[(2, 1)] is None and layout[(3, 1)] is None and layout[(1, 1)] is None and layout[(4, 2)] is None and layout[(3, 3)] is None and layout[(3, 2)] is not None and layout[(4, 1)] is None) and str(layout[(3, 2)]) == "blue_key_1": continue
                states.append(State(pos[0], pos[1], layout=layout, **curr_dict))
        
        for pos in self.positions[CellType.GOAL]:
            for values, layout in product(self._get_property_combinations(), self.layout_combinations):
                # TODO: a state where there is a key at the same position as the agent is not valid
                # obj = layout[(pos[0], pos[1])]
                # if obj is not None and obj.type == "key": continue
                terminal_states.append(State(pos[0], pos[1], layout=layout, **dict(zip(list(self.state_properties.keys()), values))))
        
        
        
        
        return states, terminal_states


    
        

    def _get_layout_combinations(self):
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
            
            all_placements.append(placement)

        # Generate all possible ways to place n-1 keys
        for key_subset in combinations(key_objects, len(key_objects) - 1):
            for key_positions in permutations(valid_positions, len(key_objects) - 1):
                placement = {pos: None for pos in valid_positions}
                
                # Doors
                for obj in door_objects:
                    placement[(obj.y, obj.x)] = obj
                
                # Keys
                for key_obj, pos in zip(key_subset, key_positions):
                    placement[pos] = key_obj
                
                all_placements.append(placement)

        return all_placements


    def __extend_properties(self, properties: dict[str, list]):
        # Do not consider key objects
        for object in self.objects:
            if object.type == "key": continue
            properties[f"{object.color}_{object.type}_{object.id}"] = [False, True]
        
        return properties

    def _get_property_combinations(self):
        property_keys = list(self.state_properties.keys())
        property_values = [self.state_properties[key] for key in property_keys]
        combinations = product(*property_values)
        
        return combinations
    
    

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
                    self.positions[CellType.WALL].append((i, j))
                elif (i, j) != self.goal_pos[0]:
                    self.positions[CellType.NORMAL].append((i, j))
                
        self.positions[CellType.START] = ((self.start_pos[0], self.start_pos[1]))
        self.positions[CellType.GOAL] = [(self.goal_pos[0][0], self.goal_pos[0][1])]
        
    
    def load_from_map(self):
        """
        Loads a grid from a predefined map and initializes cell positions.

        Raises:
        - AssertionError: If the map rows are not of equal length.
        """
        assert all(len(row) == len(self.map[0]) for row in self.map), "Not all rows have the same length"
        for j, row in enumerate(self.map):
            for i, cell in enumerate(row):
                if cell == self.char_positions[CellType.START]: self.positions[CellType.NORMAL].append((i, j))
                if cell == self.char_positions[CellType.CLIFF]: self.positions[CellType.NORMAL].append((i, j))
                self.positions[self.POSITIONS_CHAR[cell]].append((i, j))
        
        
        goal_states = self.positions[CellType.GOAL]
        self.start_pos: tuple[int, int] = self.positions[CellType.START][0]
        self.goal_pos = [(goal_state[0], goal_state[1]) for goal_state in goal_states]
        self.size_x = len(self.map)
        self.size_y = len(self.map[0])
    
    
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
    
        
    
    def is_valid(self, state: State) -> bool:
        """
        Checks if a position is valid (i.e., not a wall).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is valid, False otherwise.
        """
        return (state.y, state.x) not in self.positions[CellType.WALL]


    def is_terminal(self, state: State) -> bool:
        """
        Checks if a position is a terminal state (i.e., a goal position).

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is terminal, False otherwise.
        """
        return state in self.terminal_states
    
    
    # def terminal_state_idx(self, state: State) -> int:
    #     y, x = state.y, state.x
    #     assert (y, x) in self.goal_pos
    #     return len(self.states) + self.goal_pos.index((y, x))
    
    
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
        return (state.y, state.x) in self.positions[CellType.NORMAL] and (state.y, state.x) not in self.positions[CellType.CLIFF]
    
    
    def is_cliff(self, state: State | tuple[int, int]) -> bool:
        if type(state) == State:
            return (state.y, state.x) in self.positions[CellType.CLIFF]
        else:
            return (state[0], state[1]) in self.positions[CellType.CLIFF]
    
    
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
    