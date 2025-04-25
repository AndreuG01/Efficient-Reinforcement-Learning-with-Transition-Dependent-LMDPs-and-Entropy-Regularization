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
from utils.utils import visualize_stochasticity_rewards_embedded_lmdp, compare_value_function_by_stochasticity, lmdp_tdr_advantage, uniform_assumption_plot, generate_vi_pi_table, generate_parallel_p_table, different_gammas_plot, different_temperature_plots, regularized_embedding_error_plot, embedding_value_function_reg, embedding_errors_different_temp
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from typing import Literal
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
from math import ceil
import seaborn as sns




if __name__ == "__main__":
    embedding_value_function_reg(
        map=Map(grid_size=20),
        behaviour="stochastic",
        stochastic_prob=0.6,
        save_fig=True
    )
    # embedding_errors_different_temp(map=Map(grid_size=10), behaviour="stochastic", stochastic_prob=0.5, save_fig=True)