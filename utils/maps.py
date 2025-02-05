from .state import Object

class Maps:
    SIMPLE_TEST = [
    "####",
    "#S #",
    "# G#",
    "####"
    ]
    
    THREE = [
    "#####",
    "#S  #",
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
    
    CHALLENGE_DOOR_OBJECTS = [Object(6, 2, "red", "key", 0), Object(4, 6, "blue", "door", 0), Object(1, 4, "red", "door", 0)]
    CHALLENGE_DOOR = [
        "########",
        "#S    G#",
        "#   # C#",
        "#   #  #",
        "#   ## #",
        "#CC    #",
        "#C     #",
        "########",
    ]
    
    DOUBLE_DOOR_OBJECTS = [Object(3, 1, "green", "key", 0), Object(3, 3, "green", "door", 0), Object(4, 2, "blue", "key", 1), Object(2, 3, "blue", "door", 1)]
    DOUBLE_DOOR = [
        "#####",
        "#S#G#",
        "# # #",
        "#   #",
        "## ##",
        "#####"
    ]
    DOUBLE_KEY_OBJECTS = [Object(4, 1, "green", "key", 0), Object(4, 2, "blue", "key", 1), Object(2, 3, "green", "door", 1)]
    DOUBLE_KEY = [
        "#####",
        "#S#G#",
        "# # #",
        "#   #",
        "#  ##",
        "#####",
    ]
    
    
    