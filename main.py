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
from utils.utils import visualize_stochasticity_rewards_embedded_lmdp, compare_value_function_by_stochasticity, lmdp_tdr_advantage, uniform_assumption_plot, generate_vi_pi_table, generate_parallel_p_table, different_gammas_plot, different_temperature_plots, regularized_embedding_error_plot
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from typing import Literal
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy.stats import pearsonr
import pickle
from math import ceil
import seaborn as sns



def visualize_embedding(
    map: Map,
    f: Literal["MDP", "LMDP"] = "MDP",
    t: Literal["MDP", "LMDP"] = "LMDP",
    behaviour: Literal["deterministic", "stochastic", "mixed"] = "deterministic",
    stochastic_prob: float = 0.9,
    save_fig: bool = False
):
    assert f in ["MDP", "LMDP"]
    assert t in ["MDP", "LMDP"]
    assert t != f
    assert behaviour in ["deterministic", "stochastic", "mixed"]
    mdp_temperatures = np.arange(1, 8, 0.05)
    mdp_temperatures = [1, 2, 3, 4]
    
    
    
    lmdp_color = plt.get_cmap("Reds")
    mdp_color = plt.get_cmap("Blues")
    normalizer = Normalize(vmin=mdp_temperatures[0], vmax=mdp_temperatures[-1])
    
    sm_lmbda = cm.ScalarMappable(cmap=lmdp_color, norm=normalizer)
    sm_lmbda.set_array([])
    
    sm_beta = cm.ScalarMappable(cmap=mdp_color, norm=normalizer)
    sm_beta.set_array([])
    
    plt.rcParams.update({"text.usetex": True})
    fig, axes = plt.subplots(figsize=(10, 5))
    
    errors = {}
    lmdp_temperatures = {}
    for i, temp in enumerate(mdp_temperatures):
        curr_errors = []
        curr_temperatures = np.arange(1, 10, 0.5)
        lmdp_temperatures[temp] = curr_temperatures
        for lmdp_temp in curr_temperatures:
            print(f"MDP TEMPERATURE: {temp}. LMDP TEMPERATURE: {lmdp_temp}",)
            if f == "MDP":
                mdp = GridWorldMDP(
                    map=map,
                    allowed_actions=GridWorldActions.get_actions(),
                    behaviour=behaviour,
                    stochastic_prob=stochastic_prob,
                    temperature=temp
                )
                lmdp = mdp.to_LMDP_TDR(lmbda=lmdp_temp)
                lmdp = GridWorldLMDP_TDR(
                    map=map,
                    allowed_actions=GridWorldActions.get_actions(),
                    lmdp_tdr=lmdp
                )
            else:
                lmdp = GridWorldLMDP(
                    map=map,
                    allowed_actions=GridWorldActions.get_actions(),
                )
                mdp = lmdp.to_MDP()
                mdp = GridWorldMDP(
                    map=map,
                    allowed_actions=GridWorldActions.get_actions(),
                    mdp=mdp
                )

            mdp.compute_value_function()
            lmdp.compute_value_function()
            curr_errors.append(np.mean(np.square(mdp.V - lmdp.V)))
        errors[temp] = curr_errors
    
        
        # axes.plot([i for i in range(len(mdp.V))], mdp.V, label=f"MDP: $\\beta = {mdp.temperature}$", color=mdp_color(normalizer(temp)))
        # axes.plot([i for i in range(len(mdp.V))], lmdp.V, label=f"LMDP: $\lambda = {lmdp.lmbda}$", color=lmdp_color(normalizer(temp)))
    for beta, err in errors.items():
        temps = lmdp_temperatures[beta]
        min_err = min(err)
        min_err_idx = np.argmin(err)
        plt.plot(temps, err, label=f"MDP: $\\beta = {beta}$. Min with $\lambda = {round(temps[min_err_idx], 2)}$ ")
        plt.scatter(temps[min_err_idx], min_err, color="red", marker="x", zorder=3)
    
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("$\lambda$")
    
    # axes.set_xlabel(r"State $s$")
    # axes.set_ylabel(r"$V(s)$")
    
    # cbar_mdp = plt.colorbar(sm_beta, ax=axes)
    # cbar_mdp.set_label(f"$\\beta$")
    
    # cbar_lmdp = plt.colorbar(sm_lmbda, ax=axes)
    # cbar_lmdp.set_label(f"$\lambda$")
    
    plt.title(f"{f} to {t} embedding. Stochastic MDP with prob ${mdp.stochastic_prob if mdp.behaviour != 'deterministic' else '1'}$. Map: {map.name}")
    
    if save_fig:
        # plt.savefig(f"assets/{f}_to_{t}_prob_{mdp.stochastic_prob if mdp.behaviour != 'deterministic' else 'det'}_{map.name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"assets/{f}_to_{t}_prob_{mdp.stochastic_prob if mdp.behaviour != 'deterministic' else 'det'}_{map.name.lower().replace(' ', '_')}_lmbda_choosing.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    visualize_embedding(map=Maps.CHALLENGE_DOOR, behaviour="stochastic", stochastic_prob=0.6, save_fig=True)