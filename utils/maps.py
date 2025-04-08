from .state import Object
from typing import Optional
from .coloring import TerminalColor

class Map:
    def __init__(self, name: str, layout: list[str], objects: Optional[list[Object]] = None):
        self.name = name
        self.layout = layout
        self.objects = objects if objects else []
        self.height = len(layout)
        self.width = len(layout[0]) if layout else 0
    
    def __str__(self) -> str:
        TerminalColor.init()
        object_layer = { (obj.y, obj.x): obj.symbol() for obj in self.objects }

        col_header = "   " + "".join(f" {i:^3}" for i in range(self.width))
        separator = "   +" + "+".join(["---"] * self.width) + "+"

        lines = [f"{self.name}:", col_header, separator]

        for y in range(self.height):
            row = f"{y:2} |"
            for x in range(self.width):
                if (y, x) in object_layer:
                    symbol = object_layer[(y, x)]
                    cell = f" {symbol} "
                else:
                    char = self.layout[y][x]
                    if char == "#":
                        cell = " # "
                    elif char == "C":
                        cell = f"{TerminalColor.colorize(' C ', 'orange')}"
                    elif char in "SG":
                        cell = f" {char} "
                    else:
                        cell = "   "
                row += cell + "|"
            lines.append(row)
            lines.append(separator)

        return "\n".join(lines)

    

class Maps:
    
    EASIEST = Map(
        name="EASIEST",
        layout=[
            "###",
            "#S#",
            "#G#",
            "###",
        ]
    )
    SIMPLE_TEST = Map(
        name="SIMPLE TEST",
        layout=[
            "####",
            "#SC#",
            "# G#",
            "####"
        ]
    )
    
    THREE = Map(
        name="THREE",
        layout=[
            "#####",
            "#S# #",
            "#   #",
            "#  G#",
            "#####"
        ]
    )

    LARGE_TEST = Map(
        name="LARGE TEST",
        layout=[
            "##########",
            "#    #  G#",
            "#    #   #",
            "#    #   #",
            "#    #   #",
            "#        #",
            "#S       #",
            "##########",
        ]
    )
    
    LARGE_TEST_CLIFF = Map(
        name="LARGE TEST CLIFF",
        layout=[
            "##########",
            "#    #  G#",
            "#    #   #",
            "#    #   #",
            "#    #   #",
            "#      C #",
            "#S       #",
            "##########",
        ]
    )

    WALL_TEST = Map(
        name="WALL TEST",
        layout=[
            "##################",
            "#    ########    #",
            "#    #      #  # #",
            "#    #  #   #  # #",
            "#    #  #   #  # #",
            "#    #  #   #  # #",
            "#S      #      #G#",
            "##################"
        ]
    )
    
    CLIFF = Map(
        name="CLIFF",
        layout=[    
            "#################",
            "#               #",
            "# #CCCCCCCCCCC  #",
            "# ############  #",
            "# #   #   #  # C#",
            "# # # # # #  #  #",
            "# # # # # #  #C #",
            "# # # # # #  #  #",
            "# # # # # #  # C#",
            "# # # # # #  #  #",
            "# # # # # #  #C #",
            "# # # # # #  #  #",
            "# # # # # #  # C#",
            "# # # # # #  #  #",
            "# # # # # #  #C #",
            "#S  #   #      G#",
            "#################",
        ]
    )


    CLIFF_WALKING = Map(
        name="CLIFF WALKING",
        layout=[ # From Sutton and Barto, 2018
            "##############",
            "#            #",
            "#            #",
            "#            #",
            "#SCCCCCCCCCCG#",
            "##############",
        ]
    )
    
    CHEATING_CLIFF = Map(
        name="CHEATING CLIFF",
        layout=[
            "#######",
            "#    C#",
            "#G    #",
            "#S    #",
            "#######"
        ]
    )
    
    
    SIMPLE_DOOR = Map(
        name="SIMPLE DOOR",
        layout=[
            "#####",
            "#S###",
            "#  G#",
            "#####",
        ],
        objects=[
            Object(2, 2, "blue", "door", 0),
            Object(2, 1, "blue", "key", 0)
        ]
    )
    
    CHALLENGE_DOOR = Map(
        name="CHALLENGE DOOR",
        layout=[
            "########",
            "#S    G#",
            "#   # C#",
            "#   #  #",
            "#   ####",
            "#CC    #",
            "#C     #",
            "########",
        ],
        objects=[
            Object(6, 2, "red", "key", 0),
            Object(1, 4, "red", "door", 0)
        ]
    )
    
    
    
    DOUBLE_DOOR = Map(
        name="DOUBLE DOOR",
        layout=[
            "#####",
            "#S#G#",
            "# # #",
            "#   #",
            "#  ##",
            "#####"
        ],
        objects=[
            Object(3, 1, "green", "key", 0),
            Object(3, 2, "green", "door", 0),
            Object(2, 1, "red", "key", 1),
            Object(2, 3, "blue", "door", 1),
            Object(4, 2, "blue", "key", 1)
        ]
    )
    
    DOUBLE_KEY = Map(
        name="DOUBLE KEY",
        layout=[
            "#####",
            "#S#G#",
            "# # #",
            "#   #",
            "#  ##",
            "#####",
        ],
        objects=[
            Object(4, 1, "yellow", "key", 0),
            Object(4, 2, "yellow", "key", 1),
            Object(2, 3, "yellow", "door", 1)
        ]
    )
    
    EXAMPLE = Map(
        name="EXAMPLE",
        layout=[
            "########",
            "#S   #G#",
            "#C ### #",
            "#C     #",
            "########",
        ],
        objects=[
            Object(3, 2, "blue", "key", 0),
            Object(1, 3, "blue", "door", 0),
            Object(1, 4, "red", "key", 1),
            Object(3, 6, "red", "door", 1)
        ]
    )