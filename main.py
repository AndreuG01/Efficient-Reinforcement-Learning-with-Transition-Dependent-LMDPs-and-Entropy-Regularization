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
    
    mdp = GridWorldMDP(
        map=Maps.CLIFF,
        deterministic=True
    )
    
    lmdp = GridWorldLMDP(
        map=Maps.CLIFF,
        sparse_optimization=False
    )
    
    lmdp.compute_value_function()
    mdp.compute_value_function()
    
    fig = plt.figure(figsize=(10, 5))
    
    plt.plot([i for i in range(len(mdp.V))], mdp.V, label="MDP")
    plt.plot([i for i in range(len(lmdp.V))], lmdp.V, label="LMDP")
    
    plt.xlabel("State index")
    plt.ylabel("V(s)")
    plt.legend()
    
    plt.show()
    
    plotter = GridWorldPlotter(
        lmdp
    )
    
    plotter.plot_grid_world()