from domains.grid import MinigridActions
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP
from domains.minigrid_env import MinigridMDP, MinigridLMDP, MinigridLMDP_TDR
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
            MinigridActions.FORWARD,
            MinigridActions.PICKUP,
            MinigridActions.DROP,
            MinigridActions.TOGGLE
        ],
        map=Maps.SIMPLE_TEST,
        # objects=Maps.DOUBLE_DOOR_OBJECTS,
        deterministic=False
    )
    
    print(mdp.R)
    
    embedded_lmdp = mdp.to_LMDP()
    embedded_lmdptdr = mdp.to_LMDP_TDR()
    embedded_lmdptdr_2 = mdp.to_LMDP_TDR_2()
    
    mdp.compute_value_function()
    embedded_lmdp.compute_value_function()
    embedded_lmdptdr.compute_value_function()
    embedded_lmdptdr_2.compute_value_function()
    
    print("REWARD 1")
    print(embedded_lmdptdr.R)
    print("REWARD 2")
    print(embedded_lmdptdr_2.R)
    
    
    # lmdp = MinigridLMDP(
    #     allowed_actions=[
    #         MinigridActions.ROTATE_LEFT,
    #         MinigridActions.ROTATE_RIGHT,
    #         MinigridActions.FORWARD,
    #         # MinigridActions.PICKUP,
    #         # MinigridActions.DROP,
    #         # MinigridActions.TOGGLE
    #     ],
    #     map=Maps.SIMPLE_TEST,
    #     # objects=Maps.CHALLENGE_DOOR_OBJECTS,
    #     lmdp=embedded_lmdp
    # )
    
    # lmdp.visualize_policy(num_times=1, save_gif=True, save_path="assets/now.gif")
    
    close = np.isclose(embedded_lmdp.V, embedded_lmdptdr.V)
    if not np.all(close):
        diff_indices = np.where(~close)
        print("VALUE FUNCTIONS are NOT close at indices:")
        for idx in diff_indices[0]:
            diff = abs(embedded_lmdp.V[idx] - embedded_lmdptdr.V[idx])
            # print(f"Index ({idx}{', cliff' if idx in cliff_states else ''}): Diff = {round(diff, 4)} lmdp.V = {lmdp.V[idx]}, lmdp_tdr.V = {lmdp_tdr.V[idx]}")
            print(f"Index ({idx}): Diff = {round(diff, 4)} elmdp.V = {embedded_lmdp.V[idx]}, lmdp_tdr.V = {embedded_lmdptdr.V[idx]}.")
    else:
        print("\t\tVALUE FUNCTIONS are close", True)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(mdp.V)), mdp.V, label="MDP")
    plt.plot(np.arange(len(embedded_lmdp.V)), embedded_lmdp.V, label="Embedded LMDP")
    plt.plot(np.arange(len(embedded_lmdptdr.V)), embedded_lmdptdr.V, label="Embedded LMDP-TDR")
    plt.plot(np.arange(len(embedded_lmdptdr_2.V)), embedded_lmdptdr_2.V, label="Embedded LMDP-TDR-version 2")
    plt.xlabel("State index")
    plt.ylabel("V(s)")
    plt.title("DOUBLE_DOOR")
    
    plt.legend()
    plt.savefig("assets/mdp_lmdp-tdr.png", dpi=300)