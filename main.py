from domains.grid import MinigridActions
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP, GridWorldLMDP_TDR
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
from utils.utils import visualize_stochasticity_rewards_embedded_lmdp, compare_value_function_by_stochasticity, lmdp_tdr_advantage, uniform_assumption_plot
from sklearn.metrics import r2_score
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
import seaborn as sns

if __name__ == "__main__":
    stochastic_prob = 0.5
    fig = plt.figure(figsize=(10, 5))
    for type in ["mixed", "deterministic", "stochastic"]:
        print(type)
        mdp = MinigridMDP(
            grid_size=20,
            map=Maps.SIMPLE_DOOR,
            objects=Maps.SIMPLE_DOOR_OBJECTS,
            allowed_actions=[
                MinigridActions.ROTATE_LEFT,
                MinigridActions.ROTATE_RIGHT,
                MinigridActions.FORWARD,
                MinigridActions.PICKUP,
                MinigridActions.DROP,
                MinigridActions.TOGGLE
            ],
            behaviour=type,
            stochastic_prob=stochastic_prob
        )
        mdp.compute_value_function()
        if type == "mixed":
            label = f"Mixed (p = {stochastic_prob} for manipulation actions)"
        elif type == "deterministic":
            label = "Deterministic"
        else:
            label = f"Stochastic (p = {stochastic_prob})"
        
        plt.plot([i for i in range(len(mdp.V))], mdp.V, label=label)
    
    plt.title(f"SIMPLE DOOR MAP")
    plt.xlabel("State index")
    plt.ylabel("V(s)")
    plt.legend()
    plt.show()
    
    
    
    