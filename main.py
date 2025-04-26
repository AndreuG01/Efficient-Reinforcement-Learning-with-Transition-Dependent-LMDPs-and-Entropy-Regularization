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

def explore_temperature(map: Map, mdp_temperature: float, probs: list[float], save_fig: bool = True):
    for prob in probs:
        mdp = GridWorldMDP(
            map=map,
            allowed_actions=GridWorldActions.get_actions()[:4],
            temperature=mdp_temperature,
            behaviour="stochastic",
            stochastic_prob=prob,
            verbose=False
        )
        
        lmdp = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
        
        mdp.compute_value_function()
        
        plt.rcParams.update({"text.usetex": True})
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].plot(mdp.V, label="MDP", color="black", zorder=3, linewidth=1.5, linestyle="dashed")
        
        # Bias starting lambda based on stochastic probability
        bias = (prob - 0.5) * 2  # from -1 to 1

        start_lmbda = mdp.temperature * (1 - 0.5 * bias)
        start_lmbda = max(0.05, start_lmbda)  # ensure it's positive

        step = 0.25
        lmdp_lmbdas = [start_lmbda]
        errors = []
        value_functions = []

        # Scale how far right to extend based on how deterministic it is
        # At prob=1 → extend 80 steps; at prob=0 → extend 10 steps
        right_extension_steps = int(10 + 70 * prob)

        # Dynamic search
        step = 0
        while True:
            curr_lmbda = lmdp_lmbdas[-1]
            lmdp.compute_value_function(temp=curr_lmbda)
            value_functions.append(lmdp.V)
            curr_error = np.mean(np.square(mdp.V - lmdp.V))
            errors.append(curr_error)

            if len(errors) > 1 and curr_error > errors[-2]:
                for _ in range(right_extension_steps):
                    next_lmbda = lmdp_lmbdas[-1] + step
                    lmdp.compute_value_function(temp=next_lmbda)
                    value_functions.append(lmdp.V)
                    lmdp_lmbdas.append(next_lmbda)
                    errors.append(np.mean(np.square(mdp.V - lmdp.V)))
                break
            
            if len(value_functions) > 2:
                if np.linalg.norm(value_functions[-1] - value_functions[-2], np.inf) < 1e-3:
                    break
            
            lmdp_lmbdas.append(curr_lmbda + step)
            
            if step % 100 == 0:
                print(f"Stochastic prob: {mdp.stochastic_prob}. (mdp temperature: {mdp_temperature}, lmdp_temperature: {lmdp_lmbdas[-1]}). Iteration {step}. Last error {errors[-1]}")
                if len(value_functions) > 2: print(f"VF difference between iter {step - 1} and {step - 2}: {np.linalg.norm(value_functions[-1] - value_functions[-2], np.inf)}")
            
            step += 1

        cmap = plt.get_cmap("rainbow")
        normalizer = Normalize(vmin=min(lmdp_lmbdas), vmax=max(lmdp_lmbdas))
        sm = cm.ScalarMappable(cmap=cmap, norm=normalizer)

        for i, curr_lmbda in enumerate(lmdp_lmbdas):
            axes[0].plot(value_functions[i], color=cmap(normalizer(curr_lmbda)))

        cbar = fig.colorbar(sm, ax=axes[0])
        cbar.set_label("$\lambda$")
        
        axes[0].set_title("Value Functions by $\lambda$")

        best_lmbda = lmdp_lmbdas[np.argmin(errors)]
        best_error = min(errors)
        lmdp.compute_value_function(temp=best_lmbda)
        axes[0].plot(lmdp.V, label=f"Best LMDP ($\lambda = {round(best_lmbda, 2)})$", color="gray", zorder=3, linewidth=1.5, linestyle="dashed")

        for i, curr_lmbda in enumerate(lmdp_lmbdas):
            if curr_lmbda == best_lmbda:
                axes[1].scatter(curr_lmbda, errors[i], color=cmap(normalizer(curr_lmbda)), s=20, edgecolor="black", zorder=3)
            else:
                axes[1].scatter(curr_lmbda, errors[i], color=cmap(normalizer(curr_lmbda)), s=10)

        axes[1].axvline(x=best_lmbda, color="gray", linestyle="dashed", linewidth=1, label=f"Best $\lambda = {round(best_lmbda, 2)}$")
        axes[1].axhline(y=best_error, color="gray", linestyle="dashed", linewidth=1, label=f"MSE = {round(best_error, 4)}")

        axes[1].set_title("Error vs. $\lambda$")
        axes[1].set_xlabel("$\lambda$")
        axes[1].set_ylabel("Mean Squared Error")
        axes[1].legend(loc="upper right")

        axes[0].set_xlabel("State $s$")
        axes[0].set_ylabel("$V(s)$")
        axes[0].legend(loc="lower left")
        
        plt.suptitle(f"MDP with $\\beta = {mdp.temperature}$. Stochastic probability ${mdp.stochastic_prob if mdp.behaviour != 'deterministic' else '1'}$. LMDP with $\lambda = {lmdp.lmbda}$.\nMap: {map.name}")

        if save_fig:
            directory_1 = "assets/temperature_exploration/"
            if not os.path.exists(directory_1):
                os.mkdir(directory_1)
            
            save_map_name = map.name.lower().replace(" ", "_")
            directory_2 = os.path.join(directory_1, save_map_name)
            
            if not os.path.exists(directory_2):
                os.mkdir(directory_2)
            
            plt.savefig(os.path.join(directory_2, f"temp_{mdp.temperature}_prob_{mdp.stochastic_prob if mdp.behaviour != 'deterministic' else '1'}.png"), dpi=300, bbox_inches="tight")
    
    if not save_fig:
        plt.show()


if __name__ == "__main__":
    for map in [Maps.CLIFF]:
        for temperature in np.arange(1, 5, 0.5):
            explore_temperature(map, temperature, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    