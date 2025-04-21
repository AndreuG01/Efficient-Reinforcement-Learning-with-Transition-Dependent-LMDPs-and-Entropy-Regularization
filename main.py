from domains.grid import MinigridActions, GridWorldActions
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP, GridWorldLMDP_TDR
from domains.minigrid_env import MinigridMDP, MinigridLMDP, MinigridLMDP_TDR
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
import os
import numpy as np
from utils.maps import Maps, Map
from utils.benchmarks import benchmark_value_iteration, benchmark_parallel_p, benchmark_lmdp2mdp_embedding, benchmark_mdp2lmdp_embedding
from minigrid.manual_control import ManualControl
from custom_palette import CustomPalette
import pickle as pkl
from utils.utils import visualize_stochasticity_rewards_embedded_lmdp, compare_value_function_by_stochasticity, lmdp_tdr_advantage, uniform_assumption_plot, generate_vi_pi_table, generate_parallel_p_table, different_gammas_plot, different_temperature_plots
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
from math import ceil
import seaborn as sns

if __name__ == "__main__":
    map = Maps.CLIFF_WALKING
    
    temperature = 2.1
    mdp = GridWorldMDP(
        map=map,
        allowed_actions=GridWorldActions.get_actions()[:4],
        behaviour="deterministic",
        temperature=temperature,
    )
    
    embedded_lmdp = mdp.to_LMDP_TDR()
    
    mdp.compute_value_function(temp=embedded_lmdp.lmbda)

    embedded_lmdp.compute_value_function()
    
    print(embedded_lmdp.R)
    print(mdp.R)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(len(mdp.V))], mdp.V, label="MDP")
    plt.plot([i for i in range(len(mdp.V))], embedded_lmdp.V, label="EMBEDDED LMDP")
    plt.legend()
    plt.show()
    