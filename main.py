from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP
from domains.minigrid_env import MinigridMDP, MinigridActions, MinigridLMDP, MinigridLMDP_TDR
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import numpy as np
from utils.maps import Maps
from utils.benchmarks import benchmark_value_iteration, benchmark_parallel_p, benchmark_lmdp2mdp_embedding, benchmark_mdp2lmdp_embedding
from minigrid.manual_control import ManualControl
from custom_palette import CustomPalette
import pickle as pkl
from utils.utils import visualize_stochasticity_rewards_embedded_lmdp, compare_value_function_by_stochasticity

if __name__ == "__main__":
    
    mdp = MinigridMDP(
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD
        ],
        map=Maps.SIMPLE_TEST,
        deterministic=False
    )
    
    
    
    lmdp_tdr = MinigridLMDP_TDR(
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            MinigridActions.PICKUP,
            MinigridActions.DROP,
            MinigridActions.TOGGLE
        ],
        map=Maps.SIMPLE_DOOR,
        objects=Maps.SIMPLE_DOOR_OBJECTS,
        sparse_optimization=False,
        threads=4
    )

    lmdp = MinigridLMDP(
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            MinigridActions.PICKUP,
            MinigridActions.DROP,
            MinigridActions.TOGGLE
        ],
        map=Maps.SIMPLE_DOOR,
        objects=Maps.SIMPLE_DOOR_OBJECTS,
        sparse_optimization=True,
        threads=4
    )
    
    
    
    cliff_states = [i for i in range(lmdp.num_states) if lmdp.minigrid_env.custom_grid.is_cliff(lmdp.minigrid_env.custom_grid.state_index_mapper[i])]
    # lmdp_tdr.visualize_policy(num_times=3)
    lmdp_tdr.compute_value_function()
    lmdp.compute_value_function()
    
    print(lmdp_tdr.V)
    print(lmdp_tdr.policy)
    
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(lmdp_tdr.V)), lmdp_tdr.V, label="lmdp_tdr")
    plt.scatter(cliff_states, lmdp_tdr.V[cliff_states], marker="x", color="purple")
    plt.plot(np.arange(len(lmdp.V)), lmdp.V, label="lmdp")
    plt.scatter(cliff_states, lmdp.V[cliff_states], marker="x", color="purple", label="Cliff states")
    plt.legend()
    plt.savefig("assets/tdr_optimized.png", dpi=300)
    # plt.show()
    
    # lmdp_tdr.visualize_policy(num_times=3, save_gif=True, save_path="assets/tdr.gif")
    
    