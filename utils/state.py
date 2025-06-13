from typing import Literal, Dict, Tuple
from .coloring import TerminalColor

class Object:
    """
    Class to represent an object in the environment, such as a door or a key.
    Each object has a position (y, x), a color, a type (door or key), and an identifier.
    
    Attributes:
        y (int): The y-coordinate of the object's position.
        x (int): The x-coordinate of the object's position.
        color (str): The color of the object, one of "red", "green", "blue", "purple", "yellow", or "grey".
        type (str): The type of the object, either "door" or "key".
        id (int): A unique identifier for the object.
    """
    def __init__(
        self,
        y: int,
        x: int,
        color: Literal["red", "green", "blue", "purple", "yellow", "grey"], 
        type: Literal["door", "key"], id: int
    ):
        """
        Initializes an Object with the given position, color, type, and identifier.
        Args:
            y (int): The y-coordinate of the object's position.
            x (int): The x-coordinate of the object's position.
            color (str): The color of the object, one of "red", "green", "blue", "purple", "yellow", or "grey".
            type (str): The type of the object, either "door" or "key".
            id (int): A unique identifier for the object.
        """
        self.y = y
        self.x = x
        self.color = color
        self.type = type
        self.id = id
    
    def symbol(self) -> str:
        """
        Returns a string representation of the object symbol, which is the first letter of its type in uppercase, colorized.
        """
        return TerminalColor.colorize(self.type[0].upper(), self.color)

    
    def __str__(self):
        """
        Returns a string representation of the Object.
        """
        return f"{self.color}_{self.type}_{self.id}"
    
    def __repr__(self):
        """
        Returns a detailed string representation of the Object. Used for debugging and logging.
        """
        return f"{self.color}_{self.type}_{self.id}"

    def __eq__(self, other):
        """
        Checks if two Object instances are equal by comparing their coordinates, color, type, and id.
        
        Args:
            other (Object): The other Object instance to compare with.
        """
        if not isinstance(other, Object):
            return False
        return (self.y == other.y and
                self.x == other.x and
                self.color == other.color and
                self.type == other.type and
                self.id == other.id)

    def __hash__(self):
        """
        Returns a hash value for the Object based on its coordinates, color, type, and id.
        """
        return hash((self.y, self.x, self.color, self.type, self.id))

class State:
    """
    Class to represent a state in the state space of the reinforcement learning agents.
    It includes the position of the agent, the layout of objects in the environment, and any additional properties, such as whether the agent is carrying an object or a door is opened or closed.

    Attributes:
        y (int): The y-coordinate of the agent's position.
        x (int): The x-coordinate of the agent's position.
        layout (Dict[Tuple[int, int], Object]): A dictionary mapping coordinates to objects in the environment.
        properties (Dict[str, any]): Additional properties of the state, such as whether the agent is carrying an object or if a door is opened or closed.
    """
    def __init__(self, y: int, x: int, layout: Dict[Tuple[int, int], Object], **properties):
        """
        Initializes a State object with the given coordinates, layout, and properties.
        Args:
            y (int): The y-coordinate of the agent's position.
            x (int): The x-coordinate of the agent's position.
            layout (Dict[Tuple[int, int], Object]): A dictionary mapping coordinates to objects in the environment.
            **properties: Additional properties of the state, such as whether the agent is carrying an object or if a door is opened or closed.
        """
        self.y = y
        self.x = x
        self.layout = layout if layout is not None else {}  # Ensure layout is always a dictionary
        self.properties = properties

    def __repr__(self):
        """
        Returns a string representation of the State object, including its coordinates, layout, and properties. Used for debugging and logging.
        """
        return f"State(x={self.x}, y={self.y}, layout={self.layout}, properties={self.properties})"

    def __eq__(self, other):
        """
        Checks if two State objects are equal by comparing their coordinates, layout, and properties.
        
        Args:
            other (State): The other State instance to compare with.
        """
        if not isinstance(other, State):
            return False
        return (self.x == other.x and 
                self.y == other.y and 
                self.properties == other.properties and 
                self.layout == other.layout)

    def __hash__(self):
        """
        Returns a hash value for the State object based on its coordinates, layout, and properties.
        """
        return hash((self.x, self.y, frozenset(self.layout.items()), frozenset(self.properties.items())))
