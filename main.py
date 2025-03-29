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
    
    
    gridworld_mdp = GridWorldMDP(
        map=Maps.CLIFF,
        deterministic=True,
        stochastic_prob=0.8
    )
    
    gridworld_mdp.compute_value_function()
    print(gridworld_mdp.V)
    
    plotter = GridWorldPlotter(
        gridworld_mdp
    )
    
    plotter.plot_grid_world(
        show_value_function=True,
        show_prob=True
    )
    
    
    plt.show()
    