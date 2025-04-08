from typing import Literal, Dict, Tuple
from .coloring import TerminalColor

class Object:
    def __init__(self, y: int, x: int, color: Literal["red", "green", "blue", "purple", "yellow", "grey"], 
                 type: Literal["door", "key"], id: int):
        self.y = y
        self.x = x
        self.color = color
        self.type = type
        self.id = id
    
    def symbol(self) -> str:
        return TerminalColor.colorize(self.type[0].upper(), self.color)

    
    def __str__(self):
        return f"{self.color}_{self.type}_{self.id}"
    
    def __repr__(self):
        return f"{self.color}_{self.type}_{self.id}"

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        return (self.y == other.y and
                self.x == other.x and
                self.color == other.color and
                self.type == other.type and
                self.id == other.id)

    def __hash__(self):
        return hash((self.y, self.x, self.color, self.type, self.id))

class State:
    def __init__(self, y: int, x: int, layout: Dict[Tuple[int, int], Object], **properties):
        self.y = y
        self.x = x
        self.layout = layout if layout is not None else {}  # Ensure layout is always a dictionary
        self.properties = properties

    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, layout={self.layout}, properties={self.properties})"

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self.x == other.x and 
                self.y == other.y and 
                self.properties == other.properties and 
                self.layout == other.layout)

    def __hash__(self):
        return hash((self.x, self.y, frozenset(self.layout.items()), frozenset(self.properties.items())))
