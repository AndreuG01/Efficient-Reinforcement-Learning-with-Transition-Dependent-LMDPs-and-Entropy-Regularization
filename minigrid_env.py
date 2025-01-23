from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from grid import CustomGrid, CellType
from maps import Maps
from MDP import MDP


class CustomMinigridEnv(MiniGridEnv):
    def __init__(
        self,
        map:list[str] = None,
        grid_size: int = 3,
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        
        self.custom_grid = CustomGrid(map=map, grid_size=grid_size)
        self.agent_start_pos = self.custom_grid.start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * grid_size ** 2

        super().__init__(
            mission_space=mission_space,
            # grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            width=self.custom_grid.size_y,
            height=self.custom_grid.size_x,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Generate the walls
        for pos in self.custom_grid.positions[CellType.WALL]:
            self.grid.set(pos[0], pos[1], Wall())
        
        for goal_pos in self.custom_grid.goal_pos:
            self.put_obj(Goal(), goal_pos[0], goal_pos[1])
        
        for cliff_pos in self.custom_grid.positions[CellType.CLIFF]:
            self.put_obj(Lava(), cliff_pos[0], cliff_pos[1])
        
        

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


class MinigridMDP(MDP):
    # TODO: implement
    pass


def main():
    env = CustomMinigridEnv(render_mode="human", map=Maps.CLIFF)
    print(env.action_space)
    for action in env.actions:
        print(action)
    print(env.actions)

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    
if __name__ == "__main__":
    main()