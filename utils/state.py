class State:
    def __init__(self, y: int, x: int, **properties):
        self.y = y
        self.x = x
        self.properties = properties
    
    def __repr__(self):
        return f"State(x={self.x}, y={self.y}, properties={self.properties})"

    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y and self.properties == other.properties

    def __hash__(self):
        return hash((self.x, self.y, frozenset(self.properties.items())))