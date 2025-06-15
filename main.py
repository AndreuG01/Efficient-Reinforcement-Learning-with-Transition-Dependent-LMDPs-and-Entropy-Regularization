import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1" # To hide the welcome message of the pygame library
from models import MDP, LMDP, LMDP_TDR
from domains.grid import MinigridActions, GridWorldActions
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP, GridWorldLMDP_TDR
from domains.minigrid_env import MinigridMDP, MinigridLMDP, MinigridLMDP_TDR
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
from utils.maps import Maps, Map
from utils.benchmarks import benchmark_value_iteration, benchmark_parallel_p, benchmark_lmdp2mdp_embedding, benchmark_mdp2lmdp_embedding, benchmark_iterative_vectorized_embedding
from custom_palette import CustomPalette
from utils.experiments import visualize_stochasticity_rewards_embedded_lmdp, compare_value_function_by_stochasticity, lmdp_tdr_advantage, uniform_assumption_plot, generate_vi_pi_table, generate_parallel_p_table, different_gammas_plot, different_temperature_plots, regularized_embedding_error_plot, embedding_value_function_reg, embedding_errors_different_temp, policies_comparison, mdp_er_motivation
from typing import Literal
import time
from tqdm import tqdm

def explore_temperature(map: Map, mdp_temperature: float, probs: list[float], save_fig: bool = True):
    for prob in probs:
        mdp = MinigridMDP(
            map=map,
            # allowed_actions=GridWorldActions.get_actions(),
            allowed_actions=MinigridActions.get_actions()[:3],
            temperature=mdp_temperature,
            behavior="stochastic",
            stochastic_prob=prob,
            verbose=True
        )
        
        mdp.compute_value_function()
        
        plt.rcParams.update({"text.usetex": True})
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        
        axes[0].plot(mdp.V, label="MDP", color="black", zorder=3, linewidth=1, linestyle="dashed")
        
        # Bias starting lambda based on stochastic probability
        bias = (prob - 0.5) * 2  # from -1 to 1

        start_lmbda = mdp.temperature * (1 - 0.5 * bias)
        start_lmbda = max(0.05, start_lmbda)  # ensure it's positive
        # start_lmbda = 1

        step = 0.25
        lmdp_lmbdas = [start_lmbda]
        errors = []
        value_functions = []

        # Scale how far right to extend based on how deterministic it is
        # At prob=1 → extend 80 steps; at prob=0 → extend 10 steps

        # Dynamic search
        step = 0
        while True:
            curr_lmbda = lmdp_lmbdas[-1]
            print("Curr lambda", curr_lmbda)
            lmdp, _, _ = mdp.to_LMDP_TDR(curr_lmbda, find_best_lmbda=False)
            lmdp.compute_value_function(temp=curr_lmbda)
            value_functions.append(lmdp.V)
            curr_error = np.mean(np.square(mdp.V - lmdp.V))
            errors.append(curr_error)

            if len(errors) > 1 and curr_error > errors[-2]:
                break
                for _ in range(right_extension_steps):
                    next_lmbda = lmdp_lmbdas[-1] + step
                    lmdp, _, _ = mdp.to_LMDP_TDR(next_lmbda)
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

        cmap = plt.get_cmap("winter")
        normalizer = Normalize(vmin=min(lmdp_lmbdas), vmax=max(lmdp_lmbdas))
        sm = cm.ScalarMappable(cmap=cmap, norm=normalizer)


        cbar = fig.colorbar(sm, ax=axes[0])
        cbar.set_label("$\lambda$")
        
        axes[0].set_title("Value Functions by $\lambda$")

        best_lmbda = lmdp_lmbdas[np.argmin(errors)]
        best_error = min(errors)
        lmdp, _, _ = mdp.to_LMDP_TDR(lmbda=best_lmbda, find_best_lmbda=False)
        lmdp.compute_value_function()
        # axes[0].plot(lmdp.V, label=fr"Best LMDP ($\lambda \approx {round(best_lmbda, 2)})$", color="red", zorder=3, linewidth=1, linestyle="dashed")
        for i, curr_lmbda in enumerate(lmdp_lmbdas):
            if curr_lmbda == best_lmbda: continue
            axes[0].plot(value_functions[i], color=cmap(normalizer(curr_lmbda)))

        for i, curr_lmbda in enumerate(lmdp_lmbdas):
            if curr_lmbda == best_lmbda:
                axes[1].scatter(curr_lmbda, errors[i], color=cmap(normalizer(curr_lmbda)), s=20, edgecolor="black", zorder=3)
            else:
                axes[1].scatter(curr_lmbda, errors[i], color=cmap(normalizer(curr_lmbda)), s=10)

        axes[1].axvline(x=best_lmbda, color="gray", linestyle="dashed", linewidth=0.5, label=rf"Best $\lambda \approx {round(best_lmbda, 2)}$")
        axes[1].axhline(y=best_error, color="gray", linestyle="dashed", linewidth=0.5, label=rf"MSE $\approx {round(best_error, 4)}$")

        axes[1].set_title("Error vs. $\lambda$")
        axes[1].set_xlabel("$\lambda$")
        axes[1].set_ylabel("Mean Squared Error")
        axes[1].legend(loc="upper right")

        axes[0].set_xlabel("State $s$")
        axes[0].set_ylabel("$V(s)$")
        axes[0].legend(loc="lower left")
        
        plt.suptitle(f"MDP with $\\beta = {mdp.temperature}$. Stochastic probability ${mdp.stochastic_prob if mdp.behavior != 'deterministic' else '1'}$. LMDP with $\lambda = {round(best_lmbda, 3)}$.\nMap: {map.name}")

        if save_fig:
            directory_1 = "assets/temperature_exploration/"
            if not os.path.exists(directory_1):
                os.mkdir(directory_1)
            
            save_map_name = map.name.lower().replace(" ", "_")
            directory_2 = os.path.join(directory_1, save_map_name)
            
            if not os.path.exists(directory_2):
                os.mkdir(directory_2)
            
            plt.savefig(os.path.join(directory_2, f"temp_{mdp.temperature}_prob_{mdp.stochastic_prob if mdp.behavior != 'deterministic' else '1'}_v2.png"), dpi=300, bbox_inches="tight")
    
    if not save_fig:
        plt.show()


def mse_lambda(map: Map, beta: float, p: float, save_fig: bool = True):
    mdp = MinigridMDP(
        map=map,
        allowed_actions=MinigridActions.get_actions()[:3],
        behavior="stochastic",
        stochastic_prob=p,
        temperature=beta,
        # verbose=False
    )
    
    mdp.compute_value_function()
    mdp_v = mdp.V
    
    palette = CustomPalette()
    
    lambdas = np.arange(beta, beta + 145, 1)
    errors = []
    for lmbda in tqdm(lambdas, total=len(lambdas)):
        lmdp, _, _ = mdp.to_LMDP_TDR(lmbda=lmbda, find_best_lmbda=False)
        lmdp.compute_value_function()
        errors.append(np.mean(np.square(mdp_v - lmdp.V)))
        
    
    plt.rcParams.update({"text.usetex": True})
    fig = plt.figure(figsize=(5, 5))
    plt.plot(lambdas, errors, color=palette[16])
    
    plt.xlabel("$\lambda$")
    plt.ylabel("MSE")
    
    if save_fig:
        plt.savefig(f"assets/lmbdas_temp-{beta}_p-{p}_{map.name.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def draw_maps_initial_state():
    """
    Plots the initial state of all the maps + 2 simple grids for the report
    """
    dest_path = os.path.join("assets", "maps")
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    
    for map in [Map(grid_size=25)] + Maps.get_maps() + [Map(grid_size=4)]:
        map.R = None
        mdp_minigrid = MinigridMDP(
            map=map,
            allowed_actions=MinigridActions.get_actions()[:3],
            threads=1
        )
        
        mdp_gridworld = GridWorldMDP(
            map=map,
            allowed_actions=GridWorldActions.get_actions()[:4],
        )
        
        
        plotter = GridWorldPlotter(
            mdp_gridworld,
            name="maps",
        )
        plotter.plot_state(mdp_gridworld.environment.custom_grid.state_index_mapper[mdp_gridworld.s0], save_fig=True, fig_name=f"{map.name.replace(' ', '_')}_gridworld.png")
        
        mdp_minigrid.environment.visualize_state(mdp_minigrid.environment.custom_grid.state_index_mapper[mdp_minigrid.s0], save_path=f"{os.path.join(dest_path, map.name.replace(' ', '_'))}_minigrid.png")
        


if __name__ == "__main__":
    
    lmdp_tdr_advantage(save_fig=True)
    
    
    
    
    exit()
    # mdp_2 = MinigridMDP(
    #     map=Map(grid_size=4),
    #     allowed_actions=MinigridActions.get_actions(),
    #     stochastic_prob=0.1,
    #     behavior="stochastic",
    #     temperature=0,
    #     verbose=True,
    #     threads=1
    # )
    
    # mdp_1.compute_value_function()
    # mdp_2.compute_value_function()
    # plt.figure(figsize=(10, 5))
    # plt.plot(mdp_1.V, label=f"{mdp_1.temperature}")
    # plt.plot(mdp_2.V, label=f"{mdp_2.temperature}")
    # plt.legend()
    # plt.show()
    
    # mdp_1.visualize_policy(policies=[(0, mdp_1.policy)], num_times=100, show_window=False)#, save_gif=True, save_path="mdp1.gif")
    # mdp_2.visualize_policy(policies=[(0, mdp_2.policy)], num_times=100, show_window=False)#, save_gif=True, save_path="mdp2.gif")
    

    mdp = MinigridMDP(
        map=Map(grid_size=4),
        allowed_actions=MinigridActions.get_actions()[:3],
        stochastic_prob=0.2,
        behavior="stochastic",
        temperature=2,
        verbose=True,
        threads=1
    )
    
    # explore_temperature(map=Map(grid_size=25), mdp_temperature=1, probs=[0.6], save_fig=True)
    policies_comparison(mdp, save_fig=True)
    
    
    exit()
    
    
    
    mdp = MinigridMDP(
        map=Map(grid_size=25),
        # allowed_actions=GridWorldActions.get_actions(),
        allowed_actions=MinigridActions.get_actions()[:3],
        temperature=1.0,
        behavior="stochastic",
        stochastic_prob=0.6,
        verbose=True
    )
    # mdp.to_LMDP_TDR()
    
    # explore_temperature(Map(grid_size=20), 1.0, [0.4])
    explore_temperature(map=Map(grid_size=25), mdp_temperature=1, probs=[0.6], save_fig=False)