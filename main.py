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
        deterministic=True
    )
    print(mdp.R)
    
    
    
    lmdp_tdr = MinigridLMDP_TDR(
        grid_size=40,
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            MinigridActions.PICKUP,
            MinigridActions.DROP,
            MinigridActions.TOGGLE
        ],
        map=Maps.SIMPLE_TEST,
        # objects=Maps.CHALLENGE_DOOR_OBJECTS,
        sparse_optimization=False,
        threads=4
    )
    
    # print(lmdp_tdr.P)
    print(lmdp_tdr.R)
    # print(lmdp_tdr.P[np.where(lmdp_tdr.R == 0)])

    lmdp = MinigridLMDP(
        grid_size=40,
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            MinigridActions.PICKUP,
            MinigridActions.DROP,
            MinigridActions.TOGGLE
        ],
        map=Maps.SIMPLE_TEST,
        # objects=Maps.CHALLENGE_DOOR_OBJECTS,
        sparse_optimization=False,
        threads=4
    )
    
    
    cliff_states = [i for i in range(lmdp.num_states) if lmdp.minigrid_env.custom_grid.is_cliff(lmdp.minigrid_env.custom_grid.state_index_mapper[i])]
    print(f"Cliff states: {cliff_states}")
    # lmdp_tdr.visualize_policy(num_times=3)
    lmdp_tdr.compute_value_function()
    lmdp.compute_value_function()
    mdp.compute_value_function()
    state = 3
    # print(lmdp_tdr.policy[state, :])
    # print(lmdp.policy[state, :])

    close = np.isclose(lmdp.policy, lmdp_tdr.policy)
    if not np.all(close):
        diff_indices = np.argwhere(~close)
        print("POLICIES are NOT close at indices:")
        for idx in diff_indices:
            i, j = idx
            diff = abs(lmdp.policy[i, j] - lmdp_tdr.policy[i, j])
            print(f"Index ({i}, {j}): Diff = {round(diff, 4)}. lmdp.policy = {lmdp.policy[i, j]}, lmdp_tdr.policy = {lmdp_tdr.policy[i, j]}")
    else:
        print("\t\tPOLICIES are close", True)

    


    # close = np.isclose(lmdp.V, lmdp_tdr.V)
    # if not np.all(close):
    #     diff_indices = np.where(~close)
    #     print("VALUE FUNCTIONS are NOT close at indices:")
    #     for idx in diff_indices[0]:
    #         diff = abs(lmdp.V[idx] - lmdp_tdr.V[idx])
    #         print(f"Index ({idx}): Diff = {round(diff, 4)} lmdp.V = {lmdp.V[idx]}, lmdp_tdr.V = {lmdp_tdr.V[idx]}")
    # else:
    #     print("\t\tVALUE FUNCTIONS are close", True)

    # print(np.argmax(lmdp_tdr.policy, axis=1))
    # print(np.argmax(lmdp.policy, axis=1))
    
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(lmdp_tdr.V)), lmdp_tdr.V, label="lmdp_tdr", linewidth=1)
    plt.scatter(cliff_states, lmdp_tdr.V[cliff_states], marker="x", color="purple", s=10)
    plt.plot(np.arange(len(lmdp.V)), lmdp.V, label="lmdp", linewidth=1)
    plt.scatter(cliff_states, lmdp.V[cliff_states], marker="x", color="purple", label="Cliff states", s=10)
    
    # plt.plot(np.arange(len(mdp.V)), mdp.V, label="mdp", linewidth=1)
    # plt.scatter(cliff_states, mdp.V[cliff_states], marker="x", color="purple", s=10)
    
    # plt.plot(np.arange(len(embedded_mdp.V)), embedded_mdp.V, label="embedded_mdp", linewidth=1)
    # plt.scatter(cliff_states, embedded_mdp.V[cliff_states], marker="x", color="purple", s=10)
    plt.legend()
    plt.title("CHALLENGE DOOR")
    plt.xlabel("State index")
    plt.ylabel("V(s)")
    plt.savefig("assets/lmdp_tdr.png", dpi=300)
    # plt.show()
    
    # lmdp_tdr.visualize_policy(num_times=3, save_gif=True, save_path="assets/tdr.gif")
    
    