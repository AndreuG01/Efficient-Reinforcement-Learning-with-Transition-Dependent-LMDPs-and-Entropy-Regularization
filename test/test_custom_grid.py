# From the root directory (/Repo) run this file as python -m test.test_custom_grid

import unittest
from domains.grid import CustomGrid, MinigridActions, GridWorldActions, State, Object
from utils.maps import Map, Maps
from copy import deepcopy


class CustomGridTester(unittest.TestCase):
    def setUp(self):
        self.gridworld_grid = CustomGrid(
            type="gridworld",
            map=Maps.TESTING_MAP,
            properties={},
            allowed_actions=GridWorldActions.get_actions(),
            verbose=True
        )
        
        self.minigrid_grid = CustomGrid(
            type="mingrid",
            map=Maps.TESTING_MAP,
            properties={"orientation": [i for i in range(4)]},
            allowed_actions=MinigridActions.get_actions(),
            verbose=True
        )
        
    def assert_state_transition(self, grid, origin_state, action, expected_state, valid, terminal):
        next_state, is_valid, is_terminal = grid.move(origin_state, action)
        self.assertIn(origin_state, grid.states if origin_state not in grid.terminal_states else grid.terminal_states)
        self.assertIn(next_state, grid.states if not terminal else grid.terminal_states)
        self.assertEqual(next_state, expected_state, "Origin and destination state should be the same but they are not")
        self.assertEqual(is_valid, valid)
        self.assertEqual(is_terminal, terminal)

    def test_num_states(self):
        # Gridworld
        self.assertEqual(self.gridworld_grid.max_state_space_size, 252)
        self.assertEqual(self.gridworld_grid.get_num_states(), 219)
        
        # Minigrid
        self.assertEqual(self.minigrid_grid.max_state_space_size, 1008)
        self.assertEqual(self.minigrid_grid.get_num_states(), 876)
    
    
    """
    ##############################################################################
    #                                   RIGHT GRIDOWLD                           #
    ##############################################################################
    """
    def test_action_right(self):
        # Only for GridWorld
        origin_state_1 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": False}
        )
        self.assert_state_transition(self.gridworld_grid, origin_state_1, GridWorldActions.RIGHT, origin_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": True}
        )
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.x += 1
        self.assert_state_transition(self.gridworld_grid, origin_state_2, GridWorldActions.RIGHT, destination_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=1, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": True}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.x += 1
        self.assert_state_transition(self.gridworld_grid, origin_state_3, GridWorldActions.RIGHT, destination_state_3, True, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=2, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        destination_state_4 = deepcopy(origin_state_4)
        destination_state_4.x += 1
        self.assert_state_transition(self.gridworld_grid, origin_state_4, GridWorldActions.RIGHT, destination_state_4, True, False)
        # ========================================================================================================================================================================================================
        origin_state_5 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True}
        )
        
        destination_state_5 = deepcopy(origin_state_5)
        destination_state_5.x += 1
        self.assert_state_transition(self.gridworld_grid, origin_state_5, GridWorldActions.RIGHT, destination_state_5, True, True)

    
    """
    ##############################################################################
    #                                   LEFT GRIDWORLD                           #
    ##############################################################################
    """
    def test_action_left(self):
        origin_state_1 = State(
            y=4, x=4,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_1, GridWorldActions.LEFT, origin_state_1, True, True)
        # ========================================================================================================================================================================================================
        origin_state_2 = deepcopy(origin_state_1)
        origin_state_2.properties["purple_door_1"] = True
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.x -= 1
        
        self.assert_state_transition(self.gridworld_grid, origin_state_2, GridWorldActions.LEFT, destination_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_3, GridWorldActions.LEFT, origin_state_3, False, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=1, x=3,
            layout={
                (1, 1): None,
                (1, 2): Object(2, 3, "purple", "key", 0),
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_4, GridWorldActions.LEFT, origin_state_4, True, False)
        
        
    """
    ##############################################################################
    #                                   UP GRIDWORLD                             #
    ##############################################################################
    """
    def test_action_up(self):
        # Only for GridWorld
        origin_state_1 = State(
            y=1, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_1, GridWorldActions.UP, origin_state_1, False, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=3, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): Object(2, 3, "purple", "key", 0),
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_2, GridWorldActions.UP, origin_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): Object(2, 3, "purple", "key", 0),
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_3, GridWorldActions.UP, origin_state_3, False, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=2, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): Object(2, 3, "purple", "key", 0),
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True}
        )
        
        destination_state_4 = deepcopy(origin_state_4)
        destination_state_4.y -= 1
        
        self.assert_state_transition(self.gridworld_grid, origin_state_4, GridWorldActions.UP, destination_state_4, True, False)
        
        
    """
    ##############################################################################
    #                                   DOWN GRIDWORLD                           #
    ##############################################################################
    """
    def test_action_down(self):
        # Only for GridWorld
        origin_state_1 = State(
            y=2, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): Object(2, 3, "purple", "key", 0),
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_1, GridWorldActions.DOWN, origin_state_1, False, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_2, GridWorldActions.DOWN, origin_state_2, False, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=1, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.y += 1
        
        self.assert_state_transition(self.gridworld_grid, origin_state_3, GridWorldActions.DOWN, destination_state_3, True, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=4, x=4,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )

        self.assert_state_transition(self.gridworld_grid, origin_state_4, GridWorldActions.DOWN, origin_state_4, False, True)
    
    """
    ##############################################################################
    #                                  FORWARD MINIGRID                          #
    ##############################################################################
    """
    def test_action_forward(self):
        # Only for MiniGrid
        origin_state_1 = State(
            y=1, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False, "orientation": 2}
        )
        
        destination_state_1 = deepcopy(origin_state_1)
        destination_state_1.y = 1
        destination_state_1.x = 1
        
        self.assert_state_transition(self.minigrid_grid, origin_state_1, MinigridActions.FORWARD, destination_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=4, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False, "orientation": 1}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_2, MinigridActions.FORWARD, origin_state_2, False, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 0}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.x = 4

        self.assert_state_transition(self.minigrid_grid, origin_state_3, MinigridActions.FORWARD, destination_state_3, True, True)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=1, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": True, "orientation": 1}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_4, MinigridActions.FORWARD, origin_state_4, True, False)
        # ========================================================================================================================================================================================================
        origin_state_5 = State(
            y=1, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": True, "orientation": 0}
        )
        
        destination_state_5 = deepcopy(origin_state_5)
        destination_state_5.x = 4
        
        self.assert_state_transition(self.minigrid_grid, origin_state_5, MinigridActions.FORWARD, destination_state_5, True, False)
        
    """
    ##############################################################################
    #                              ROTATE LEFT MINIGRID                          #
    ##############################################################################
    """
    def test_action_rotate_left(self):
        # Only for MiniGrid
        origin_state_1 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): Object(2, 3, "purple", "key", 0),
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False, "orientation": 0}
        )
        
        destination_state_1 = deepcopy(origin_state_1)
        destination_state_1.properties["orientation"] = 3
        
        self.assert_state_transition(self.minigrid_grid, origin_state_1, MinigridActions.ROTATE_LEFT, destination_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=2, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): Object(2, 3, "purple", "key", 0),
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False, "orientation": 3}
        )
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.properties["orientation"] = 2
        
        self.assert_state_transition(self.minigrid_grid, origin_state_2, MinigridActions.ROTATE_LEFT, destination_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=1, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): Object(2, 3, "purple", "key", 0),
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False, "orientation": 2}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.properties["orientation"] = 1
        
        self.assert_state_transition(self.minigrid_grid, origin_state_3, MinigridActions.ROTATE_LEFT, destination_state_3, True, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=4, x=4,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): Object(2, 3, "purple", "key", 0),
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 1}
        )
        
        destination_state_4 = deepcopy(origin_state_4)
        destination_state_4.properties["orientation"] = 0
        
        self.assert_state_transition(self.minigrid_grid, origin_state_4, MinigridActions.ROTATE_LEFT, destination_state_4, True, True)
        
        
    """
    ##############################################################################
    #                             ROTATE RIGHT MINIGRID                          #
    ##############################################################################
    """
    def test_action_rotate_right(self):
        # Only for MiniGrid
        origin_state_1 = State(
            y=1, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 3}
        )
        
        destination_state_1 = deepcopy(origin_state_1)
        destination_state_1.properties["orientation"] = 0
        
        self.assert_state_transition(self.minigrid_grid, origin_state_1, MinigridActions.ROTATE_RIGHT, destination_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=4, x=4,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 0}
        )
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.properties["orientation"] = 1
        
        self.assert_state_transition(self.minigrid_grid, origin_state_2, MinigridActions.ROTATE_RIGHT, destination_state_2, True, True)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=3, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 1}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.properties["orientation"] = 2
        self.assert_state_transition(self.minigrid_grid, origin_state_3, MinigridActions.ROTATE_RIGHT, destination_state_3, True, False)
        # ========================================================================================================================================================================================================
        
        origin_state_4 = State(
            y=1, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 2}
        )
        
        destination_state_4 = deepcopy(origin_state_4)
        destination_state_4.properties["orientation"] = 3
        
        self.assert_state_transition(self.minigrid_grid, origin_state_4, MinigridActions.ROTATE_RIGHT, destination_state_4, True, False)
        
    """
    ##############################################################################
    #                                   PICKUP GRIDWORLD                         #
    ##############################################################################
    """
    def pickup_gridworld(self):
        origin_state_1 = State(
            y=1, x=3,
            layout={
                (1, 1): None,
                (1, 2): Object(2, 3, "purple", "key", 0),
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        destination_state_1 = deepcopy(origin_state_1)
        destination_state_1.layout[(1, 2)] = None
        self.assert_state_transition(self.gridworld_grid, origin_state_1, GridWorldActions.PICKUP, destination_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=1, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_2, GridWorldActions.PICKUP, origin_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): Object(2, 3, "purple", "key", 0),
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.layout[(4, 1)] = None
        
        self.assert_state_transition(self.gridworld_grid, origin_state_3, GridWorldActions.PICKUP, destination_state_3, True, False)
        # ========================================================================================================================================================================================================
        double_key = CustomGrid(
            type="gridworld",
            map=Maps.DOUBLE_KEY,
            properties={},
            allowed_actions=GridWorldActions.get_actions(),
            verbose=True
        )
        
        origin_state_4 = State(
            y=3, x=2,
            layout={
                (2, 1): None,
                (3, 1): None,
                (1, 1): None,
                (4, 2): Object(4, 2, "yellow", "key", 1),
                (3, 3): None,
                (3, 2): None,
                (4, 1): Object(4, 1, "yellow", "key", 0),
                (2, 3): Object(2, 3, "yellow", "door", 2)
            },
            **{'yellow_door_2': False}
        )
        
        destination_state_4 = deepcopy(origin_state_4)
        destination_state_4.layout[(4, 2)] = None
        
        self.assert_state_transition(double_key, origin_state_4, GridWorldActions.PICKUP, destination_state_4, True, False)
        # ========================================================================================================================================================================================================
        origin_state_5 = State(
            y=3, x=2,
            layout={
                (2, 1): None,
                (3, 1): None,
                (1, 1): None,
                (4, 2): Object(4, 2, "yellow", "key", 1),
                (3, 3): None,
                (3, 2): None,
                (4, 1): None,
                (2, 3): Object(2, 3, "yellow", "door", 2)
            },
            **{'yellow_door_2': False}
        )
        
        self.assert_state_transition(double_key, origin_state_5, GridWorldActions.PICKUP, origin_state_5, True, False)
        # ========================================================================================================================================================================================================
        origin_state_6 = State(
            y=3, x=2,
            layout={
                (2, 1): None,
                (3, 1): Object(4, 2, "yellow", "key", 1),
                (1, 1): None,
                (4, 2): None,
                (3, 3): Object(4, 1, "yellow", "key", 0),
                (3, 2): None,
                (4, 1): None,
                (2, 3): Object(2, 3, "yellow", "door", 2)
            },
            **{'yellow_door_2': False}
        )
        
        destination_state_6 = deepcopy(origin_state_6)
        destination_state_6.layout[(3, 3)] = None
        
        self.assert_state_transition(double_key, origin_state_6, GridWorldActions.PICKUP, destination_state_6, True, False)
    
    """
    ##############################################################################
    #                                   PICKUP MINIGRID                          #
    ##############################################################################
    """
    def pickup_minigrid(self):
        origin_state_1 = State(
            y=1, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": True, "orientation": 1}
        )
        
        destination_state_1 = deepcopy(origin_state_1)
        destination_state_1.layout[(2, 3)] = None
        
        self.assert_state_transition(self.minigrid_grid, origin_state_1, MinigridActions.PICKUP, destination_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = deepcopy(origin_state_1)
        origin_state_2.properties["orientation"] = 3
        
        self.assert_state_transition(self.minigrid_grid, origin_state_2, MinigridActions.PICKUP, origin_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): Object(2, 3, "purple", "key", 0),
            },
            **{"purple_door_1": False, "orientation": 0}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_3, MinigridActions.PICKUP, origin_state_3, True, False)
        # ========================================================================================================================================================================================================
        double_key = CustomGrid(
            type="minigrid",
            map=Maps.DOUBLE_KEY,
            properties={"orientation": [i for i in range(4)]},
            allowed_actions=MinigridActions.get_actions(),
            verbose=True
        )
        
        origin_state_4 = State(
            y=3, x=2,
            layout={
                (2, 1): None,
                (3, 1): None,
                (1, 1): None,
                (4, 2): Object(4, 2, "yellow", "key", 1),
                (3, 3): None,
                (3, 2): None,
                (4, 1): None,#Object(4, 1, "yellow", "key", 0),
                (2, 3): Object(2, 3, "yellow", "door", 2)
            },
            **{'yellow_door_2': False, "orientation": 1}
        )
        
        self.assert_state_transition(double_key, origin_state_4, MinigridActions.PICKUP, origin_state_4, True, False)
        # ========================================================================================================================================================================================================
        origin_state_5 = State(
            y=3, x=2,
            layout={
                (2, 1): None,
                (3, 1): Object(4, 2, "yellow", "key", 1),
                (1, 1): None,
                (4, 2): None,
                (3, 3): Object(4, 1, "yellow", "key", 0),
                (3, 2): None,
                (4, 1): None,
                (2, 3): Object(2, 3, "yellow", "door", 2)
            },
            **{'yellow_door_2': False, "orientation": 0}
        )
        
        destination_state_5 = deepcopy(origin_state_5)
        destination_state_5.layout[(3, 3)] = None
        
        self.assert_state_transition(double_key, origin_state_5, MinigridActions.PICKUP, destination_state_5, True, False)
    
    
    def test_action_pickup(self):
        # Both GridWorld and MiniGrid
        self.pickup_gridworld()
        self.pickup_minigrid()
    
    
    """
    ##############################################################################
    #                                   DROP GRIDWORLD                           #
    ##############################################################################
    """
    def drop_gridworld(self):
        origin_state_1 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): Object(2, 3, "purple", "key", 0),
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        self.assert_state_transition(self.gridworld_grid, origin_state_1, GridWorldActions.DROP, origin_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.layout[(4, 1)] = Object(2, 3, "purple", "key", 0)
        
        self.assert_state_transition(self.gridworld_grid, origin_state_2, GridWorldActions.DROP, destination_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=1, x=1,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.layout[(1, 2)] = Object(2, 3, "purple", "key", 0)
        
        self.assert_state_transition(self.gridworld_grid, origin_state_3, GridWorldActions.DROP, destination_state_3, True, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True}
        )
        
        destination_state_4 = deepcopy(origin_state_4)
        destination_state_4.layout[(4, 2)] = Object(2, 3, "purple", "key", 0)
        
        self.assert_state_transition(self.gridworld_grid, origin_state_4, GridWorldActions.DROP, destination_state_4, True, False)
        
    
    """
    ##############################################################################
    #                                   DROP MINIGRID                            #
    ##############################################################################
    """
    def drop_minigrid(self):
        origin_state_1 = State(
            y=2, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False, "orientation": 0}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_1, MinigridActions.DROP, origin_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=2, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False, "orientation": 3}
        )
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.layout[(1, 3)] = Object(2, 3, "purple", "key", 0)
        
        self.assert_state_transition(self.minigrid_grid, origin_state_2, MinigridActions.DROP, destination_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 0}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_3, MinigridActions.DROP, origin_state_3, True, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 2}
        )
        
        destination_state_4 = deepcopy(origin_state_4)
        destination_state_4.layout[(4, 2)] = Object(2, 3, "purple", "key", 0)
        
        self.assert_state_transition(self.minigrid_grid, origin_state_4, MinigridActions.DROP, destination_state_4, True, False)
        
        
    def test_action_drop(self):
        # Both GridWorld and MiniGrid
        self.drop_gridworld()
        self.drop_minigrid()
    
    """
    ##############################################################################
    #                                TOGGLE GRIDWORLD                            #
    ##############################################################################
    """
    def toggle_gridworld(self):
        origin_state_1 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True}
        )
    
        self.assert_state_transition(self.gridworld_grid, origin_state_1, GridWorldActions.TOGGLE, origin_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True}
        )
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.properties["purple_door_1"] = False
        
        self.assert_state_transition(self.gridworld_grid, origin_state_2, GridWorldActions.TOGGLE, destination_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": False}
        )
        
        destination_state_3 = deepcopy(origin_state_3)
        destination_state_3.properties["purple_door_1"] = True
        
        self.assert_state_transition(self.gridworld_grid, origin_state_3, GridWorldActions.TOGGLE, destination_state_3, True, False)
    
    """
    ##############################################################################
    #                                   TOGGLE MINIGRID                          #
    ##############################################################################
    """
    def toggle_minigrid(self):
        origin_state_1 = State(
            y=4, x=3,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 2}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_1, MinigridActions.TOGGLE, origin_state_1, True, False)
        # ========================================================================================================================================================================================================
        origin_state_2 = State(
            y=4, x=2,
            layout={
                (1, 1): None,
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 0}
        )
        
        destination_state_2 = deepcopy(origin_state_2)
        destination_state_2.properties["purple_door_1"] = False
        
        self.assert_state_transition(self.minigrid_grid, origin_state_2, MinigridActions.TOGGLE, destination_state_2, True, False)
        # ========================================================================================================================================================================================================
        origin_state_3 = State(
            y=4, x=2,
            layout={
                (1, 1): Object(2, 3, "purple", "key", 0),
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 0}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_3, MinigridActions.TOGGLE, origin_state_3, True, False)
        # ========================================================================================================================================================================================================
        origin_state_4 = State(
            y=4, x=4,
            layout={
                (1, 1): Object(2, 3, "purple", "key", 0),
                (1, 2): None,
                (1, 3): None,
                (2, 1): None,
                (4, 3): Object(4, 3, "purple", "door", 1),
                (3, 1): None,
                (4, 1): None,
                (4, 2): None,
                (2, 3): None,
            },
            **{"purple_door_1": True, "orientation": 2}
        )
        
        self.assert_state_transition(self.minigrid_grid, origin_state_4, MinigridActions.TOGGLE, origin_state_4, True, True)
        
    
    def test_action_toggle(self):
        # Both GridWorld and MiniGrid
        self.toggle_gridworld()
        self.toggle_minigrid()
        
        

if __name__ == "__main__":
    unittest.main()