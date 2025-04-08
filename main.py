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
from scipy.stats import spearmanr
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
import seaborn as sns

if __name__ == "__main__":
    
    size = 2
    mdp = GridWorldMDP(
        grid_size=size,
        map=Maps.CLIFF,
        # behaviour="stochastic",
        # behaviour="deterministic",
        # stochastic_prob=0.8
    )
    
    # mdp.to_LMDP_TDR_3()
    
    embedded_lmdp = mdp.to_LMDP()
    
    mdp.compute_value_function()
    embedded_lmdp.compute_value_function()
    
    palette = CustomPalette()
    
    plt.rcParams.update({"text.usetex": True})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot([i for i in range(len(mdp.V))], mdp.V, label="MDP $\mathcal{{M}}$", color=palette[15])
    axes[0].plot([i for i in range(len(mdp.V))], embedded_lmdp.V, label="LMDP $\mathcal{{L}}$", color=palette[12])
    axes[0].set_xlabel("State index")
    axes[0].set_ylabel("$V(s)$")
    axes[0].set_title(f"MSE: {np.mean(np.square(mdp.V - embedded_lmdp.V))}")
    axes[0].legend()
    
    axes[1].scatter(mdp.V, embedded_lmdp.V, color=palette[19], linewidths=0.2, edgecolors="black")
    axes[1].plot(mdp.V, mdp.V, color="gray", linestyle="--", lw=1)
    axes[1].set_xlabel("$V_{\mathcal{M}}(s)$")
    axes[1].set_ylabel("$V_{\mathcal{L}}(s)$")
    axes[1].set_title(f"$R^2$: {r2_score(mdp.V, embedded_lmdp.V)}. Corrcoef: {np.corrcoef(mdp.V, embedded_lmdp.V)[0, 1]}.\nSpearman: {spearmanr(mdp.V, embedded_lmdp.V)[0]}")
    
    plt.suptitle(f"Stochastic prob $p = {mdp.stochastic_prob}$")
    # plt.savefig(f"assets/size_{size}_p_{str(p).replace('.', '_')}", dpi=300, bbox_inches="tight")
    plt.show()
    
    
    # lmdp = MinigridLMDP(
    #     grid_size=size,
    #     lmdp=embedded_lmdp
    # )
    # lmdp.visualize_policy()