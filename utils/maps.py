from .state import Object

class Maps:
    
    EASIEST = [
        "###",
        "#S#",
        "#G#",
        "###",
    ]
    
    SIMPLE_TEST = [
    "####",
    "#SC#",
    "# G#",
    "####"
    ]
    
    THREE = [
    "#####",
    "#S# #",
    "#   #",
    "#  G#",
    "#####"
    ]


    LARGE_TEST = [
        "##########",
        "#    #  G#",
        "#    #   #",
        "#    #   #",
        "#    #   #",
        "#        #",
        "#S       #",
        "##########",
    ]
    LARGE_TEST_CLIFF = [
        "##########",
        "#    #  G#",
        "#    #   #",
        "#    #   #",
        "#    #   #",
        "#      C #",
        "#S       #",
        "##########",
    ]

    WALL_TEST = [
        "##################",
        "#    ########    #",
        "#    #      #  # #",
        "#    #  #   #  # #",
        "#    #  #   #  # #",
        "#    #  #   #  # #",
        "#S      #      #G#",
        "##################"
    ]

    CLIFF = [
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
    
    CLIFF_2 = [
        "#######",
        "#     #",
        "#     #",
        "#SCCCG#",
        "#######",
    ]

    CHEATING_CLIFF = [
        "#######",
        "#    C#",
        "#G    #",
        "#S    #",
        "#######"
    ]
    
    SIMPLE_DOOR_OBJECTS = [Object(2, 2, "blue", "door", 0), Object(2, 1, "blue", "key", 0)]
    SIMPLE_DOOR = [
        "#####",
        "#S###",
        "#  G#",
        "#####",
        "#####",
    ]
    
    CHALLENGE_DOOR_OBJECTS = [Object(6, 2, "red", "key", 0), Object(1, 4, "red", "door", 0)]
    CHALLENGE_DOOR = [
        "########",
        "#S    G#",
        "#   # C#",
        "#   #  #",
        "#   ####",
        "#CC    #",
        "#C     #",
        "########",
    ]
    
    # DOUBLE_DOOR_OBJECTS = [Object(2, 3, "blue", "door", 1), Object(4, 2, "blue", "key", 1), Object(4, 1, "green", "key", 1)]
    DOUBLE_DOOR_OBJECTS = [Object(3, 1, "green", "key", 0), Object(3, 2, "green", "door", 0), Object(2, 1, "red", "key", 1), Object(2, 3, "blue", "door", 1), Object(4, 2, "blue", "key", 1)]
    DOUBLE_DOOR = [
        "#####",
        "#S#G#",
        "# # #",
        "#   #",
        "#  ##",
        "#####"
    ]
    DOUBLE_KEY_OBJECTS = [Object(4, 1, "yellow", "key", 0), Object(4, 2, "yellow", "key", 1), Object(2, 3, "yellow", "door", 1)]
    DOUBLE_KEY = [
        "#####",
        "#S#G#",
        "# # #",
        "#   #",
        "#  ##",
        "#####",
    ]
    
    THREE_DOOR_OBJECTS = [Object(3, 1, "blue", "key", 0), Object(3, 2, "blue", "door", 0)]
    THREE_DOOR = [
        "#####",
        "#S  #",
        "# ###",
        "#  G#",
        "#####",
    ]
    
    FOUR_DOOR_OBJECTS = [Object(4, 2, "blue", "key", 0), Object(4, 3, "blue", "door", 0)]
    FOUR_DOOR = [
        "######",
        "#S   #",
        "#    #",
        "#  ###",
        "#   G#",
        "######"
    ]
    
    
    FIVE_DOOR_OBJECTS = [Object(5, 3, "blue", "key", 0), Object(5, 4, "blue", "door", 0)]
    FIVE_DOOR = [
        "#######",
        "#S    #",
        "#     #",
        "#     #",
        "#   ###",
        "#    G#",
        "#######"
    ]
    
    SIX_DOOR_OBJECTS = [Object(6, 4, "blue", "key", 0), Object(6, 5, "blue", "door", 0)]
    SIX_DOOR = [
        "########",
        "#S     #",
        "#      #",
        "#      #",
        "#      #",
        "#    ###",
        "#     G#",
        "########"
    ]
    
    
    