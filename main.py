from domains.grid import MinigridActions, GridWorldActions
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP, GridWorldLMDP_TDR
from domains.minigrid_env import MinigridMDP, MinigridLMDP, MinigridLMDP_TDR
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import numpy as np
from utils.maps import Maps, Map
from utils.benchmarks import benchmark_value_iteration, benchmark_parallel_p, benchmark_lmdp2mdp_embedding, benchmark_mdp2lmdp_embedding
from minigrid.manual_control import ManualControl
from custom_palette import CustomPalette
import pickle as pkl
from utils.utils import visualize_stochasticity_rewards_embedded_lmdp, compare_value_function_by_stochasticity, lmdp_tdr_advantage, uniform_assumption_plot, generate_vi_pi_table, generate_parallel_p_table
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
import seaborn as sns

if __name__ == "__main__":
    generate_parallel_p_table()
    exit()
    
    # mdp = MinigridMDP(
    #     map=Map(grid_size=10),
    #     behaviour="stochastic"
    # )
    
    embedded_lmdp = mdp.to_LMDP()
    # embedded_lmdp_tdr = mdp.to_LMDP_TDR_3()
    
    mdp.compute_value_function()
    embedded_lmdp.compute_value_function()
    # embedded_lmdp_tdr.compute_value_function()
    
    print(embedded_lmdp.R)
    # print(embedded_lmdp_tdr.R)
    
    print("LMDP", np.where(embedded_lmdp.policy[0] != 0), embedded_lmdp.policy[0, np.where(embedded_lmdp.policy[0] != 0)])
    
    # for state in range(embedded_lmdp_tdr.num_non_terminal_states):
    #     print("LMDP-TDR", np.where(embedded_lmdp_tdr.policy[state] != 0), embedded_lmdp_tdr.policy[state, np.where(embedded_lmdp_tdr.policy[state] != 0)])
    
    palette = CustomPalette()
    
    plt.rcParams.update({"text.usetex": True})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot([i for i in range(len(mdp.V))], mdp.V, label=f"MDP $\mathcal{{M}}$", color=palette[16])
    axes[0].plot([i for i in range(len(mdp.V))], embedded_lmdp.V, label=f"LMDP $\mathcal{{L}}$, MSE: {round(np.mean(np.square(mdp.V - embedded_lmdp.V)), 5)}", color=palette[5])
    # axes[0].plot([i for i in range(len(mdp.V))], embedded_lmdp_tdr.V, label=f"LMDP $\mathcal{{L}}$-TDR, MSE: {round(np.mean(np.square(mdp.V - embedded_lmdp_tdr.V)), 5)}", color=palette[6])
    axes[0].set_xlabel("State index")
    axes[0].set_ylabel("$V(s)$")
    # axes[0].set_title(f"")
    axes[0].legend()
    
    axes[1].scatter(mdp.V, embedded_lmdp.V, color=palette[5], linewidths=0.2, edgecolors="black", label=f"LMDP, $R^2$: {round(r2_score(mdp.V, embedded_lmdp.V), 3)}. Corrcoef: {round(np.corrcoef(mdp.V, embedded_lmdp.V)[0, 1], 4)}. Spearman: {round(spearmanr(mdp.V, embedded_lmdp.V)[0], 4)}")
    # axes[1].scatter(mdp.V, embedded_lmdp_tdr.V, color=palette[6], linewidths=0.2, edgecolors="black", label=f"LMDP-TDR, $R^2$: {round(r2_score(mdp.V, embedded_lmdp_tdr.V), 3)}. Corrcoef: {round(np.corrcoef(mdp.V, embedded_lmdp_tdr.V)[0, 1], 4)}. Spearman: {round(spearmanr(mdp.V, embedded_lmdp_tdr.V)[0], 4)}")
    axes[1].plot(mdp.V, mdp.V, color="gray", linestyle="--", lw=1, label="Ideal")
    axes[1].set_xlabel("$V_{\mathcal{M}}(s)$")
    axes[1].set_ylabel("$V_{\mathcal{L}}(s)$")
    axes[1].set_title(f". ")
    axes[1].legend()
    
    if isinstance(mdp, GridWorldMDP):
        plt.suptitle(f"{mdp.gridworld_env.title}. Stochastic prob $p = {f'{mdp.stochastic_prob}$' if not mdp.deterministic else f'1$ (deterministic)'}")
    else:
        plt.suptitle(f"{mdp.minigrid_env.title}. Stochastic prob $p = {f'{mdp.stochastic_prob}$' if not mdp.deterministic else f'1$ (deterministic)'}")
    # plt.savefig(f"assets/size_{size}_p_{str(p).replace('.', '_')}", dpi=300, bbox_inches="tight")
    plt.show()
    
    
    # lmdp = MinigridLMDP(
    #     grid_size=size,
    #     lmdp=embedded_lmdp
    # )
    # lmdp.visualize_policy()