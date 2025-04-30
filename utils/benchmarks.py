from domains.grid_world import GridWorldMDP
from domains.minigrid_env import MinigridMDP, MinigridLMDP
from domains.grid import MinigridActions
from utils.maps import Map
from utils.state import Object
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from joblib import cpu_count
import numpy as np
from custom_palette import CustomPalette
from utils.stats import ModelBasedAlgsStats
from typing import Literal
from collections import defaultdict
from tqdm import tqdm


def benchmark_value_iteration(savefig: bool = True):
    time_efficient = []
    time_inefficient = []
    min_size = 2
    max_size = 20
    for size in range(min_size, max_size):
        gridworld_mdp = GridWorldMDP(
            map=Map(grid_size=size),
            behaviour="deterministic"
        )
        print(f"[{size} / {max_size}]. Efficient...")
        _, stats_efficient = gridworld_mdp.value_iteration()
        print(f"[{size} / {max_size}]. Inefficient...")
        _, stats_inefficient = gridworld_mdp.value_iteration_inefficient()
        
        time_efficient.append(stats_efficient.time)
        time_inefficient.append(stats_inefficient.time)
    
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({
        "text.usetex": True,
    })
    plt.plot([i for i in range(min_size, max_size)], time_efficient, marker="o", label="Efficient Value Iteration", color="blue")
    plt.plot([i for i in range(min_size, max_size)], time_inefficient, marker="o", label="Inefficient Value Iteration", color="red")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Comparison between two implementations of value iteration in a simple grid domain", fontsize=14, fontweight="bold")
    plt.xlabel("Grid size")
    plt.ylabel("Time ($s$)")
    plt.grid()
    if savefig:
        plt.savefig("assets/benchmark/value_iteration.png", dpi=300)
    else:
        plt.show()


def benchmark_parallel_p(
    savefig: bool = True,
    visual: bool = False,
    min_grid: int = 10,
    max_grid: int = 65,
    framewok: Literal["MDP", "LMDP"] = "MDP",
    behaviour: Literal["stochastic", "deterministic"] = "deterministic"
):
    
    assert framewok in ["MDP", "LMDP"], "Invalid framework"
    assert behaviour in ["stochastic", "deterministic"], "Invalid behaviour"
    
    results = defaultdict(list)
    
    for jobs in tqdm(range(1, cpu_count() // 2 + 1), desc="Benchmarking parallel P", total=cpu_count() // 2):
        for grid_size in np.arange(min_grid, max_grid + 1, 5):
            print(f"{jobs} cpus and size {grid_size}")
            if framewok == "MDP":
                model = MinigridMDP(
                    map=Map(grid_size=grid_size),
                    allowed_actions=MinigridActions.get_actions(),
                    benchmark_p=True,
                    threads=jobs,
                    behaviour=behaviour
                )
            elif framewok == "LMDP":
                model = MinigridLMDP(
                    map=Map(grid_size=grid_size),
                    allowed_actions=MinigridActions.get_actions(),
                    benchmark_p=True,
                    threads=jobs,
                    sparse_optimization=False
                )
            
            results[jobs].append([model.num_states, model.p_time])
        
    
    if not visual:
        return results
    
    palette = CustomPalette()
    
    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update({
        "text.usetex": True,
    })
    
    for i, core in enumerate(results.keys()):
        times = [elem[1] for elem in results[core]]
        states = [elem[0] for elem in results[core]]
        if i == 0:
            plt.plot(states, times, label=f"{core} core", color=palette[i], linestyle="--")
        else:
            plt.plot(states, times, label=f"{core} cores", color=palette[i], linewidth=1, marker="x")
        
    
    plt.title("Parallelization impact on the generation time of transition matrix $\mathcal{P}$.", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("Number of states")
    plt.ylabel("Time ($s$)")
    
    if savefig:
        plt.savefig(f"assets/benchmark/parallel_p_{framewok}_{behaviour}.png", dpi=300)
    else:
        plt.show()
    
    return results
        
        
def benchmark_lmdp2mdp_embedding(
    map: Map,
    savefig: bool = True,
    ):
    
    custom_palette = CustomPalette()
    lmdp_color = custom_palette[3]
    mdp_color = custom_palette[4]
    
    save_name = map.name    
    
    minigrid_lmdp = MinigridLMDP(
        map=map,
        allowed_actions=MinigridActions.get_actions(),
        sparse_optimization=True,
        threads=6
    )
    
    minigrid_lmdp.compute_value_function()
    lmdp_v = minigrid_lmdp.V
    
    states_to_goal = minigrid_lmdp.states_to_goal()
    
    embedded_mdp = minigrid_lmdp.to_MDP()
    mdp_v = embedded_mdp.V
    error_all = np.mean(np.square(lmdp_v - mdp_v))
    error_some = np.mean(np.square(lmdp_v[states_to_goal] - mdp_v[states_to_goal]))
    
    
    fig1 = plt.figure(figsize=(10, 5))
    plt.rcParams.update({
        "text.usetex": True
    })
    plt.scatter(states_to_goal, lmdp_v[states_to_goal], label="States that lead to the goal faster", color=custom_palette[0], s=8, marker="x", zorder=3)
    plt.plot([i for i in range(len(lmdp_v))], lmdp_v, label="LMDP", color=lmdp_color, linewidth=1)
    plt.plot([i for i in range(len(mdp_v))], mdp_v, label="MDP", color=mdp_color, linewidth=1)
    plt.suptitle(f"LMDP and its embedded MDP comparison. {save_name}", fontsize=14, fontweight="bold")
    plt.title(f"MSE: {error_all:,.3e}\nMSE only with States that lead to the goal faster: {error_some:,.3e}", fontsize=10)
    plt.xlabel("State")
    plt.grid()
    plt.ylabel("Value function")
    plt.legend()    
    
    stats_lmdp: ModelBasedAlgsStats = minigrid_lmdp.stats
    stats_mdp: ModelBasedAlgsStats = embedded_mdp.stats
    
    print("LMDP stats")
    stats_lmdp.print_statistics()
    
    print("MDP stats")
    stats_mdp.print_statistics()
    
    fig2 = plt.figure(figsize=(10, 5))
    plt.suptitle(f"LMDP and its embedded MDP comparison. {save_name}. {minigrid_lmdp.num_states} states", fontsize=14, fontweight="bold")
    plt.title(f"Value Iteration and Power Iteration convergence", fontsize=10)
    plt.plot([i for i in range(len(stats_lmdp.deltas))], stats_lmdp.deltas, color=lmdp_color, label=rf"Power Iteration: ${stats_lmdp.time:2f}$ sec")
    plt.plot([i for i in range(len(stats_mdp.deltas))], stats_mdp.deltas, color=mdp_color, label=rf"Value Iteration: ${stats_mdp.time:2f}$ sec")
    plt.xlabel("Iteration")
    plt.ylabel(r"$| V_k - V_{k-1}|$")
    plt.grid()
    plt.legend()
    
    if savefig:
        fig1.savefig(f"assets/benchmark/lmdp2mdp/value_function_comparison_{save_name}.png", dpi=300)
        fig2.savefig(f"assets/benchmark/lmdp2mdp/deltas_{save_name}.png", dpi=300)
    else:
        plt.show()
    
    
    stats_mdp.value_fun_evolution_gif("assets/benchmark/lmdp2mdp", f"value_function_evolution_{save_name}.gif", stats_lmdp)


def benchmark_mdp2lmdp_embedding(
    map: Map,
    savefig: bool = True,
    allowed_actions: list = None,
    visual: bool = False
) -> tuple[ModelBasedAlgsStats, ModelBasedAlgsStats]:
    custom_palette = CustomPalette()
    lmdp_color = custom_palette[3]
    mdp_color = custom_palette[4]
    
    save_name = map.name

    minigrid_mdp = MinigridMDP(
        map=map,
        allowed_actions=allowed_actions,
        behaviour="stochastic", #TODO: change when embedding for deterministic MDP is implemented
    )
    
    
    cliff_states = [state for state in range(minigrid_mdp.num_states) if minigrid_mdp.environment.custom_grid.is_cliff(minigrid_mdp.environment.custom_grid.state_index_mapper[state])]
    
    embedded_lmdp = minigrid_mdp.to_LMDP()
    
    stats_lmdp: ModelBasedAlgsStats = minigrid_mdp.stats
    stats_mdp: ModelBasedAlgsStats = embedded_lmdp.stats
    
    if not visual:
        return stats_mdp, stats_lmdp
    
    # minigrid_mdp.compute_value_function()
    mdp_v = minigrid_mdp.V
    lmdp_v = embedded_lmdp.V
    error_all = np.mean(np.square(lmdp_v - mdp_v))
    
    fig1 = plt.figure(figsize=(10, 5))
    plt.rcParams.update({
        "text.usetex": True
    })
    plt.scatter(cliff_states, lmdp_v[cliff_states], label="Cliff states", color=custom_palette[0], s=8, marker="x", zorder=3)
    plt.scatter(cliff_states, mdp_v[cliff_states], color=custom_palette[0], s=8, marker="x", zorder=3)
    plt.plot([i for i in range(len(lmdp_v))], lmdp_v, label="LMDP", color=lmdp_color, linewidth=1)
    plt.plot([i for i in range(len(mdp_v))], mdp_v, label="MDP", color=mdp_color, linewidth=1)
    plt.suptitle(f"LMDP and its embedded MDP comparison. {map.name}", fontsize=14, fontweight="bold")
    plt.title(f"MSE: {error_all:,.3e}", fontsize=10)
    plt.xlabel("State")
    plt.grid()
    plt.ylabel("Value function")
    plt.legend()    
    
    
    fig2 = plt.figure(figsize=(10, 5))
    plt.suptitle(f"LMDP and its embedded MDP comparison. {map.name}. {minigrid_mdp.num_states} states", fontsize=14, fontweight="bold")
    plt.title(f"Value Iteration and Power Iteration convergence", fontsize=10)
    plt.plot([i for i in range(len(stats_lmdp.deltas))], stats_lmdp.deltas, color=lmdp_color, label=rf"Power Iteration: ${stats_lmdp.time:2f}$ sec")
    plt.plot([i for i in range(len(stats_mdp.deltas))], stats_mdp.deltas, color=mdp_color, label=rf"Value Iteration: ${stats_mdp.time:2f}$ sec")
    plt.xlabel("Iteration")
    plt.ylabel(r"$| V_k - V_{k-1}|$")
    plt.grid()
    plt.legend()
    
    if savefig:
        fig1.savefig(f"assets/benchmark/mdp2lmdp/value_function_comparison_{save_name}.png", dpi=300)
        fig2.savefig(f"assets/benchmark/mdp2lmdp/deltas_{save_name}.png", dpi=300)
    else:
        plt.show()
    
    return stats_mdp, stats_lmdp