from .state import Object
from typing import Optional
from .coloring import TerminalColor
import numpy as np

class Map:
    def __init__(
        self,
        name: str = None,
        grid_size: int = None,
        layout: list[str] = None,
        objects: Optional[list[Object]] = None,
        R: np.ndarray = None,
        P: np.ndarray = None
    ):
        if layout is None:
            assert grid_size is not None, "If no layout is provided, either specify grid size or provide a layout"
        else:
            assert name is not None, "A name must be specified if a layout is given"
        
        if grid_size is not None:
            assert grid_size > 0, f"Grid size ({grid_size}) must be strictly greater than 0"
        
        self.name = name if name else f"{grid_size} x {grid_size} Simple Grid"
        
        self.objects = objects if objects else []
        self.grid_size = grid_size
        
        if layout:
            self.__height = len(layout)
            self.__width = len(layout[0]) if layout else 0
        else:
            self.__height = grid_size + 2
            self.__width = grid_size + 2
        
        self.layout = layout if layout else self.__create_layout()
        
        self.R = R
        self.P = P
    
    
    def __create_layout(self):
        layout = []
        for x in range(self.__width):
            curr_line = ""
            for y in range(self.__height):
                if x == 1 and y == 1:
                    curr_line += "S"
                elif x == self.grid_size and y == self.grid_size:
                    curr_line += "G"
                elif x == 0 or y == 0 or x == self.__width - 1 or y == self.__height - 1:
                    curr_line += "#"
                else:
                    curr_line += " "    
            layout.append(curr_line )
        
        return layout
    
    def __str__(self) -> str:
        TerminalColor.init()
        object_layer = { (obj.y, obj.x): obj.symbol() for obj in self.objects }

        col_header = "   " + "".join(f" {i:^3}" for i in range(self.__width))
        separator = "   +" + "+".join(["---"] * self.__width) + "+"

        lines = [f"{self.name}:", col_header, separator]

        for y in range(self.__height):
            row = f"{y:2} |"
            for x in range(self.__width):
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
    
    @classmethod
    def get_maps(cls) -> list[Map]:
        return [value for _, value in cls.__dict__.items() if type(value) == Map]
        
    
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
    
    USELESS_OBJECTS = Map(
        name="USELESS OBJECTS",
        layout=[
            "########",
            "#S  G# #",
            "#C ### #",
            "#C     #",
            "########",
        ],
        objects=[
            Object(1, 2, "purple", "key", 0),
            Object(1, 3, "yellow", "key", 0),
            Object(1, 6, "purple", "door", 0),
        ]
    )
    
    MDP_NON_UNIFORM_REWARD = Map(
        name="NONE UNIFORM REWARDS",
        layout=[
            "######",
            "#S   #",
            "#    #",
            "#C   #",
            "#   G#",
            "######",
        ],
        R = np.array([
            [-8, -5, -5, -10],
            [-8, -7, -5, -5],
            [-8, -5, -5, -5],
            [-8, -10, -5, -5],
            [-5, -5, -5, -5],
            [-5, -5, -5, -5],
            [-5, -5, -5, -5],
            [-5, -10, -5, -5],
            [-15, -15, -15, -15],
            [-5, -5, -5, -5],
            [-5, -5, -5, -5],
            [-5, -10, -5, -5],
            [-5, -5, -8, -10],
            [-5, -5, -8, -5],
            [-5, -5, -8, -5],
            [0, 0, 0, 0],
        ])
    )

    LMDP_CUSTOM_REWARDS = Map(
        name="CUSTOM REWARDS",
        layout=[
            "######",
            "#S   #",
            "#  # #",
            "# #  #",
            "#   G#",
            "######"
        ],
        R = np.array([
            [-6], [-5], [-4], [-3],
            [-5], [-4], [-2],
            [-4], [-2], [-1],
            [-3], [-2], [-1], [0]
        ]).reshape(-1)
    )
    
    LMDP_TDR_CUSTOM_REWARDS = Map(
        name="CUSTOM REWARDS",
        layout=[
            "######",
            "#S   #",
            "#  # #",
            "# #  #",
            "#   G#",
            "######"
        ],
        R = np.array([
            [-6, -5, 0, 0, -5, 0, 0, 0, 0, 0, 0, 0 ,0, 0],
            [-6, -5, -4, 0, 0, -4, 0, 0, 0, 0, 0, 0 ,0, 0],
            [0, -5, -4, -3, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0],
            [0, 0, -4, -3, 0, 0, -2, 0, 0, 0, 0, 0 ,0, 0],
            [-6, 0, 0, 0, -5, -4, 0, -4, 0, 0, 0, 0 ,0, 0],
            [0, -5, 0, 0, -5, -4, 0, 0, 0, 0, 0, 0 ,0, 0],
            [0, 0, 0, -3, 0, 0, -2, 0, 0, -1, 0, 0 ,0, 0],
            [0, 0, 0, 0, -5, 0, 0, -4, 0, 0, -3, 0 ,0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0 , -1, 0],
            [0, 0, 0, 0, 0, 0, -2, 0, -2, -1, 0, 0 ,0, 0],
            [0, 0, 0, 0, 0, 0, 0, -4, 0, 0, -3, -2 ,0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -2 ,-1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, -2, -1, 0],
        ])
    )
    
    GRIDWORLD_MDP_MYOPIC = Map(
        name="MYOPIC",
        layout=[
            "########",
            "#S     #",
            "# #### #",
            "# #    #",
            "# # ####",
            "#     G#",
            "########"
        ],
        R = np.array([
            [-10, -10, -10, -10],
            [-9, -9, -9, -9],
            [-4, -4, -4, -4],
            [-7, -7, -7, -7],
            [-6, -6, -6, -6],
            [-5, -5, -5, -5],
            [-9, -9, -9, -9],
            [-4, -4, -4, -4],
            [-5, -5, -5, -5],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-2, -2, -2, -2],
            [-3, -3, -3, -3],
            [-5, -5, -5, -5],
            [-1, -1, -1, -1],
            [-5, -5, -5, -5],
            [-5, -5, -5, -5],
            [-5, -5, -5, -5],
            [-5, -5, -5, -5],
            [-5, -5, -5, -5],
            [0, 0, 0, 0],
        ])
        # R = np.array([
        #     [-10, -10, -10, -10],
        #     [-9, -9, -9, -9],
        #     [-4, -4, -4, -4],
        #     [-7, -7, -7, -7],
        #     [-6, -6, -6, -6],
        #     [-5, -5, -5, -5],
        #     [-9, -9, -9, -9],
        #     [-4, -4, -4, -4],
        #     [-5, -5, -5, -5],
        #     [-1, -1, -1, -1],
        #     [-1, -1, -1, -1],
        #     [-2, -2, -2, -2],
        #     [-3, -3, -3, -3],
        #     [-5, -5, -5, -5],
        #     [-1, -1, -1, -1],
        #     [-5, -5, -5, -5],
        #     [-5, -5, -5, -5],
        #     [-10, -10, -10, -10],
        #     [-10, -10, -10, -10],
        #     [-10, -10, -10, -10],
        #     [0, 0, 0, 0],
        # ])
    )
    
    
    TESTING_MAP = Map(
        name="Testing map",
        layout=[
            "######",
            "#S  C#",
            "# C C#",
            "# C###",
            "#   G#",
            "######",
        ],
        objects=[
            Object(2, 3, "purple", "key", 0),
            Object(4, 3, "purple", "door", 0),
        ]
    )