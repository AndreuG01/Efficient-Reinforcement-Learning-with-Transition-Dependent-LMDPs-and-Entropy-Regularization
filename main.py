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
            # MinigridActions.PICKUP,
            # MinigridActions.DROP,
            # MinigridActions.TOGGLE
        ],
        map=Maps.LARGE_TEST_CLIFF,
        # objects=Maps.DOUBLE_DOOR_OBJECTS,
        deterministic=True,
    )
    
    
    
    lmdp = MinigridLMDP(
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            # MinigridActions.PICKUP,
            # MinigridActions.DROP,
            # MinigridActions.TOGGLE
        ],
        map=Maps.LARGE_TEST_CLIFF,
        # objects=Maps.DOUBLE_DOOR_OBJECTS,
        sparse_optimization=False,
    )
    
    lmdp_tdr = MinigridLMDP_TDR(
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            # MinigridActions.PICKUP,
            # MinigridActions.DROP,
            # MinigridActions.TOGGLE
        ],
        map=Maps.LARGE_TEST_CLIFF,
        # objects=Maps.DOUBLE_DOOR_OBJECTS,
        sparse_optimization=False
    )
    
    
    mdp.compute_value_function()
    states_goal, actions_goal = mdp.states_to_goal(include_actions=True)
    states_goal = states_goal[:-1]
    actions_goal = [int(action) for action in actions_goal[:-1]]
    
    print("States to goal: ", states_goal)
    print("Actions to goal: ", actions_goal)
    
    
    for i in range(len(states_goal) - 1):
        lmdp_tdr.R[states_goal[i], states_goal[i+1]] = -1
        lmdp.R[states_goal[i+1]] = -1
        mdp.R[states_goal[i], actions_goal[i]] = -1
        
    
    
    
    
    
    mdp.compute_value_function()
    lmdp.compute_value_function()
    lmdp_tdr.compute_value_function()
    
    
    
    
    
    
    diff_mdp_lmdp = np.linalg.norm(mdp.V - lmdp.V)
    diff_mdp_lmdptdr = np.linalg.norm(mdp.V - lmdp_tdr.V)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(len(lmdp.V))], lmdp.V, color="blue", label="LMDP")
    plt.plot([i for i in range(len(mdp.V))], mdp.V, color="green", label="MDP")
    plt.plot([i for i in range(len(lmdp_tdr.V))], lmdp_tdr.V, color="red", label="LMDP-TDR")
    plt.xlabel("State index")
    plt.ylabel("V(s)")
    plt.title(f"MDP vs LMPD: {round(diff_mdp_lmdp, 3)}. MDP vs LMDP-TDR: {round(diff_mdp_lmdptdr, 3)}")
    plt.grid()
    plt.legend()
    
    plt.show()
    