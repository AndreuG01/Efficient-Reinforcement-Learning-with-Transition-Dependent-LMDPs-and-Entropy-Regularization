from utils.state import State
from itertools import product

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
    DOOR = 5
    KEY = 6
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
        "C": CellType.CLIFF,
        "D": CellType.DOOR,
        "K": CellType.KEY
    }

    def __init__(self, map: list[str] = None, grid_size: int = 3, properties: dict[str, list] = None):
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
            
        
        if self.map is None:
            self.generate_simple_grid(grid_size=grid_size)
        else:
            self.load_from_map()
        
        self.generate_state_index_mapper()
    
    def _generate_states(self, x: int, y: int) -> list[State]:
        property_keys = list(self.state_properties.keys())
        property_values = [self.state_properties[key] for key in property_keys]
        combinations = product(*property_values)
        
        return [State(x, y, **dict(zip(property_keys, values))) for values in combinations]
        

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
                
        self.positions[CellType.START].append(State(self.start_pos[0], self.start_pos[1], **{k: v[0] for k, v in self.state_properties.items()}))
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
                if cell == self.char_positions[CellType.KEY]: self.positions[CellType.NORMAL].extend(self._generate_states(i, j))
                if cell == self.char_positions[CellType.DOOR]: self.positions[CellType.NORMAL].extend(self._generate_states(i, j))
                self.positions[self.POSITIONS_CHAR[cell]].extend(self._generate_states(i, j))
        
        start_state = self.positions[CellType.START][0]
        goal_states = self.positions[CellType.GOAL]
        self.start_pos: tuple[int, int] = (start_state.y, start_state.x)
        self.goal_pos = [(goal_state.x, goal_state.y) for goal_state in goal_states]
        self.size_x = len(self.map)
        self.size_y = len(self.map[0])
    
    
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
        return state in self.positions[CellType.KEY]
    
    
    def is_door(self, state: State) -> bool:
        """
        Checks if a position is a door.

        Args:
        - pos (tuple[int, int]): The position to check.

        Returns:
        - bool: True if the position is a door, False otherwise.
        """
        return state in self.positions[CellType.DOOR]
    
    
    def print_grid(self):
        """
        Prints a textual representation of the grid world, showing the types of cells (normal, wall, start, goal).
        """
        for j in range(self.size_x):
            for i in range(self.size_y):
                type = [k for k, v in self.grid.positions.items() if (i, j) in v][-1]
                print(self.char_positions[type], end="")
            print()
    