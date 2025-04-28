import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1" # To hide the welcome message of the pygame library
from models.MDP import MDP
from domains.grid import MinigridActions, GridWorldActions
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP, GridWorldLMDP_TDR
from domains.minigrid_env import MinigridMDP, MinigridLMDP, MinigridLMDP_TDR
from models import LMDP_TDR
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image
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
            verbose=True
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


def policies_comparison(
    mdp: GridWorldMDP | MinigridMDP,
    lmdp: LMDP_TDR,
    temp_1: float, temp_2: float,
    save_fig: bool = True,
    zoom: bool = False,
    zoom_size: int = 50
):
    def kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-10) -> float:
    
        Q_safe = np.clip(Q, epsilon, None)
        Q_safe = Q_safe / np.sum(Q_safe, axis=1, keepdims=True)
        
        return np.mean(np.sum(P * (np.log(P + epsilon) - np.log(Q_safe)), axis=1))

    def extract_diagonal_window(matrix: np.ndarray, size: int) -> np.ndarray:
        n = matrix.shape[0]
        center = n // 2
        start = max(0, center - size // 2)
        end = min(n, center + size // 2)
        return matrix[start:end, start:end]

    cmap = "jet"

    plt.rcParams.update({"text.usetex": True})

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_big = fig.add_subplot(gs[1, :])

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.85, wspace=0.3, hspace=0.4)

    if not hasattr(mdp, "policy"):
        print("HERE")
        mdp.compute_value_function()


    game_times = 1000

    mdp_policy = mdp.to_LMDP_policy().astype(np.float64)
    stats_mdp = mdp.visualize_policy(num_times=game_times, show_window=False)
    print(f"MDP STATS: {stats_mdp.GAME_INFO}")

    new_ldmp = MinigridLMDP_TDR(
        map=mdp.minigrid_env.custom_grid.map,
        allowed_actions=mdp.allowed_actions,
        lmdp=lmdp,
        verbose=False
    )

    new_ldmp.compute_value_function(temp=temp_1)
    kl_1 = kl_divergence(mdp_policy, new_ldmp.policy)
    print(f"kl_1: {kl_1}, mse1: {np.mean(np.square(mdp_policy - new_ldmp.policy))}")
    policy_1 = new_ldmp.policy
    stats_lmdp_1 = new_ldmp.visualize_policy(policies=[(0, policy_1)], num_times=game_times, show_window=False)
    print(f"LMDP STATS 1: {stats_lmdp_1.GAME_INFO}")

    new_ldmp.compute_value_function(temp=temp_2)
    kl_2 = kl_divergence(mdp_policy, new_ldmp.policy)
    print(f"kl_2: {kl_2}, mse2: {np.mean(np.square(mdp_policy - new_ldmp.policy))}")
    policy_2 = new_ldmp.policy
    stats_lmdp_2 = new_ldmp.visualize_policy(policies=[(0, policy_2)], num_times=game_times, show_window=False)
    print(f"LMDP STATS 2: {stats_lmdp_2.GAME_INFO}")

    policy_data = [
        mdp_policy,
        policy_1.astype(np.float64),
        policy_2.astype(np.float64)
    ]

    min_kl = min(kl_1, kl_2)

    if zoom:
        policy_data = [extract_diagonal_window(data, zoom_size) for data in policy_data]

    vmin = min(np.min(data) for data in policy_data if data is not None)
    vmax = max(np.max(data) for data in policy_data if data is not None)

    im1 = ax1.imshow(policy_data[0], vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title(f"$\pi_{{\mathcal{{M}}}}^{{\\beta = {temp_1}}}(s'\mid s)$")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect("equal")

    im2 = ax2.imshow(policy_data[1], vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title(f"$\pi_{{\mathcal{{L}}}}^{{\lambda = {temp_1}}}(s'\mid s)$.")
    kl_1_formatting = f"\\textbf{{{round(kl_1, 3)}}}" if kl_1 == min_kl else f"{round(kl_1, 3)}"
    ax2.set_xlabel(f"KL$\left(\pi_{{\mathcal{{M}}}} (\cdot\mid s) || \pi_{{\mathcal{{L}}}}(\cdot \mid s)\\right) = {kl_1_formatting}$")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect("equal")

    im3 = ax3.imshow(policy_data[2], vmin=vmin, vmax=vmax, cmap=cmap)
    ax3.set_title(f"$\pi_{{\mathcal{{L}}}}^{{\lambda = {temp_2}}}(s'\mid s)$.")
    kl_2_formatting = f"\\textbf{{{round(kl_2, 3)}}}" if kl_2 == min_kl else f"{round(kl_2, 3)}"
    ax3.set_xlabel(f"KL$\left(\pi_{{\mathcal{{M}}}} (\cdot\mid s) || \pi_{{\mathcal{{L}}}}(\cdot \mid s)\\right) = {kl_2_formatting}$.")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect("equal")

    
    # Bottom plot
    ax_big.plot(stats_mdp.get_proportion_correct_moves_round(), label="MDP $\mathcal{M}$", linewidth=0.5, color="blue", alpha=0.5)
    ax_big.hlines(np.mean(stats_mdp.get_proportion_correct_moves_round()), xmin=0, xmax=game_times, color="blue", linestyles="--", zorder=3)
    ax_big.plot(stats_lmdp_1.get_proportion_correct_moves_round(), label=f"LMDP $\lambda = {temp_1}$", linewidth=0.5, color="red", alpha=0.5)
    ax_big.hlines(np.mean(stats_lmdp_1.get_proportion_correct_moves_round()), xmin=0, xmax=game_times, color="red", linestyles="--", zorder=3)
    ax_big.plot(stats_lmdp_2.get_proportion_correct_moves_round(), label=f"LMDP $\lambda = {temp_2}$", linewidth=0.5, color="green", alpha=0.5)
    ax_big.hlines(np.mean(stats_lmdp_2.get_proportion_correct_moves_round()), xmin=0, xmax=game_times, color="green", linestyles="--", zorder=3)
    
    yticks_base = np.linspace(0, 1, 6) * 100
    means = [
        np.mean(stats_mdp.get_proportion_correct_moves_round()),
        np.mean(stats_lmdp_1.get_proportion_correct_moves_round()),
        np.mean(stats_lmdp_2.get_proportion_correct_moves_round())
    ]
    
    yticks = np.unique(np.concatenate((yticks_base, means)))
    yticks = np.sort(yticks)

    threshold = 5
    yticks = [yticks[0]] + [yticks[i] for i in range(1, len(yticks)) if yticks[i] - yticks[i - 1] > threshold]
    
    

    ax_big.set_yticks(yticks)
    ax_big.set_yticklabels([f"{tick:.2f}" for tick in yticks])
    
    
    ax_big.legend()
    ax_big.set_xlabel("Game number")
    ax_big.set_ylabel("Correct actions ($\%$)")
    # ax_big.set_aspect("equal")

    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax)


    plt.suptitle(f"{'Zoomed policy' if zoom else 'Policy'} comparison between MDP $\mathcal{{M}}$ and embedded LMDP $\mathcal{{L}}$ with different $\\beta = \lambda$ on the value function computation. Stochastic prob: {mdp.stochastic_prob}")

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_fig:
        save_map_name = mdp.minigrid_env.custom_grid.map.name.lower().replace(" ", "_")
        plt.savefig(f"assets/{'zoomed_' if zoom else ''}policy_comparison_{save_map_name}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()



def kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-10) -> float:
    
    Q_safe = np.clip(Q, epsilon, None)
    Q_safe = Q_safe / np.sum(Q_safe, axis=1, keepdims=True)
    
    return np.mean(np.sum(P * (np.log(P + epsilon) - np.log(Q_safe)), axis=1))


if __name__ == "__main__":
    # P = np.array([
    #     [0.7, 0.1, 0.2, 0],
    #     [0, 0.9, 0, 0.1],
    #     [0.1, 0, 0.4, 0.5]
    # ])
    
    # Q = np.array([
    #     [0.8, 0.1, 0.1, 0],
    #     [0.3, 0.7, 0, 0],
    #     [0.1, 0.1, 0.1, 0.7]
    # ])
    
    # print(kl_divergence(P, Q, epsilon=1e-100))
    
    # exit()
    
    mdp = MinigridMDP(
        map=Maps.DOUBLE_KEY,
        allowed_actions=MinigridActions.get_actions(),
        behaviour="stochastic",
        stochastic_prob=0.3,
        temperature=1,
        verbose=True,
        dtype=np.float128
    )
    
    
    lmdp = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
    
    policies_comparison(mdp, lmdp, temp_1=mdp.temperature, temp_2=92.2, save_fig=True, zoom=False)
    # policies_comparison(mdp, lmdp, temp_1=mdp.temperature, temp_2=1381.9, save_fig=True, zoom=True, zoom_size=100)
    