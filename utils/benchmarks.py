# This file contains the benchmark functions used to evaluate the performance of different implementations and algorithms.

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
from .spinner import Spinner


def benchmark_value_iteration(min_size: int = 2, max_size: int = 20, save_path: str = "assets/benchmark/value_iteration_comparison.txt", save_fig: bool = True) -> None:
    """
    Benchmark the time taken by the vectorized and iterative value iteration algorithms.
    
    The function stores the results in a text file and optionally saves figures comparing the time taken by both algorithms.
    
    Args:
        min_size (int): Minimum grid size for the benchmark.
        max_size (int): Maximum grid size for the benchmark.
        save_path (str): Path to save the benchmark results.
        save_fig (bool): Whether to save the figures or show them.
    
    Returns:
        None
    """
    time_efficient = []
    time_inefficient = []
    state_space_sizes = []
    table_lines = []
    spinner = None

    header_line = "+------------+---------------------+-----------------------+"
    table_lines.append(header_line)
    table_lines.append("| Num States | Efficient Time (s) | Inefficient Time (s) |")
    table_lines.append(header_line)

    try:
        for size in range(min_size, max_size):
            gridworld_mdp = GridWorldMDP(
                map=Map(grid_size=size),
                behavior="deterministic",
                verbose=False
            )
            spinner = Spinner(f"[{size} / {max_size}] | Efficient")
            spinner.start()
            _, stats_efficient, _ = gridworld_mdp.value_iteration()
            spinner.stop()
            
            spinner = Spinner(f"[{size} / {max_size}] | Inefficient")
            spinner.start()
            _, stats_inefficient = gridworld_mdp.value_iteration_inefficient()
            spinner.stop()
            
            efficient_time = stats_efficient.time
            inefficient_time = stats_inefficient.time
            num_states = gridworld_mdp.num_states

            time_efficient.append(efficient_time)
            time_inefficient.append(inefficient_time)
            state_space_sizes.append(num_states)

            line = f"| {num_states:<10} | {efficient_time:<19.6f} | {inefficient_time:<21.6f} |"
            table_lines.append(line)

    except KeyboardInterrupt:
        print(f"Stopping spinner threads".ljust(40))
        if spinner and spinner.running:
            spinner.stop(interrupted=True)
        exit()

    table_lines.append(header_line)
    table_str = "\n".join(table_lines)
    print(table_str)

    if save_path:
        with open(save_path, "w") as f:
            f.write(table_str)

    
    plt.rcParams.update({
        "text.usetex": True,
    })
    
    fig1 = plt.figure(figsize=(8, 5))
    plt.title("Iterative vs Vectorized Value Iteration Time Comparison", fontsize=14, fontweight="bold")
    plt.plot(state_space_sizes, time_efficient, marker="o", label="Vectorized Value Iteration", color="blue")
    plt.plot(state_space_sizes, time_inefficient, marker="o", label="Iterative Value Iteration", color="red")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Time ($s$)")
    plt.xlabel("Number of states")
    plt.grid()
    
    if save_fig:
        plt.savefig("assets/benchmark/value_iteration_time_comparison.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    palette = CustomPalette()
    
    fig2 = plt.figure(figsize=(8, 5))
    plt.title("Iterative vs Vectorized Value Iteration Speedup", fontsize=14, fontweight="bold")
    plt.ylabel("Speedup")
    plt.xlabel("Number of states")
    plt.plot(state_space_sizes, np.array(time_inefficient) / np.array(time_efficient), color=palette[16])
    plt.grid()
    
    if save_fig:
        plt.savefig("assets/benchmark/value_iteration_speedup.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    



def benchmark_parallel_p(
    savefig: bool = True,
    visual: bool = False,
    min_grid: int = 10,
    max_grid: int = 65,
    framewok: Literal["MDP", "LMDP"] = "MDP",
    behavior: Literal["stochastic", "deterministic"] = "deterministic"
) -> None | dict[int, list[list[int | float]]]:
    """
    Benchmark the parallelization of the transition matrix P generation with different numbers of CPU cores and grid sizes.
    
    Args:
        savefig (bool): Whether to save the figure or show it.
        visual (bool): Whether to visualize the results or not. If not visual, returns the results as a dictionary.
        min_grid (int): Minimum grid size for the benchmark.
        max_grid (int): Maximum grid size for the benchmark.
        framewok (Literal["MDP", "LMDP"]): The framework to use, either "MDP" or "LMDP". Defaults to "MDP".
        behavior (Literal["stochastic", "deterministic"]): The behavior of the environment, either "stochastic" or "deterministic". Defaults to "deterministic".
        
    Returns:
        None or dict[int, list[list[int | float]]]: If visual is False, returns a dictionary with the number of jobs as keys and a list of [num_states, p_time] as values. If visual is True, returns None.
    """
    
    assert framewok in ["MDP", "LMDP"], "Invalid framework"
    assert behavior in ["stochastic", "deterministic"], "Invalid behavior"
    
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
                    behavior=behavior
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
        plt.savefig(f"assets/benchmark/parallel_p_{framewok}_{behavior}.png", dpi=300)
    else:
        plt.show()
    
    return results
        
        
def benchmark_lmdp2mdp_embedding(
    map: Map,
    savefig: bool = True,
) -> None:
    """
    Benchmark the embedding of a LMDP into an MDP.
    
    Args:
        map (Map): The map to use for the benchmark.
        savefig (bool): Whether to save the figures or show them.
        
    Returns:
        None
    """
    
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
    """
    Benchmark the embedding of an MDP into a LMDP.
    
    Args:
        map (Map): The map to use for the benchmark.
        savefig (bool): Whether to save the figures or show them.
        allowed_actions (list): List of allowed actions for the MDP.
        visual (bool): Whether to visualize the results or not. If not visual, returns the stats.
    
    Returns:
        tuple[ModelBasedAlgsStats, ModelBasedAlgsStats]: The stats of the MDP and LMDP.
    """
    custom_palette = CustomPalette()
    lmdp_color = custom_palette[3]
    mdp_color = custom_palette[4]
    
    save_name = map.name

    minigrid_mdp = MinigridMDP(
        map=map,
        allowed_actions=allowed_actions,
        behavior="stochastic",
    )
    
    
    cliff_states = [state for state in range(minigrid_mdp.num_states) if minigrid_mdp.environment.custom_grid.is_cliff(minigrid_mdp.environment.custom_grid.state_index_mapper[state])]
    
    embedded_lmdp, _ = minigrid_mdp.to_LMDP()
    
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


def benchmark_iterative_vectorized_embedding(max_grid_size: int = 60, save_path="assets/benchmark/iterative_vs_vectorized_embedding.txt") -> None:
    """
    Benchmark the time taken by the vectorized and iterative MDP to LMDP_TDR embedding.
    It stores the results in a text file and plots the speedup and absolute time difference.
    
    Args:
        max_grid_size (int): Maximum grid size for the benchmark.
        save_path (str): Path to save the benchmark results.
    
    Returns:
        None
    """
    table_lines = []
    header_line = "+------------+-----------------------+-----------------------+"
    table_lines.append(header_line)
    table_lines.append("| Num States | Vectorized Time (s)  | Iterative Time (s)    |")
    table_lines.append(header_line)
    
    sizes = np.arange(3, max_grid_size, 2)
    spinner = None
    vectorized_times = []
    iterative_times = []
    try:
        for grid_size in sizes:
            mdp = MinigridMDP(
                map=Map(grid_size=grid_size),
                allowed_actions=MinigridActions.get_actions(),
                behavior="stochastic",
                stochastic_prob=0.3,
                temperature=4.5,
                verbose=False
            )
            mdp.compute_value_function()
            
            
            spinner = Spinner(f"Grid size: {grid_size:>2} | Vectorized")
            spinner.start()
            _, vectorized_stats, _ = mdp.to_LMDP_TDR(find_best_lmbda=True, vectorized=True)
            spinner.stop()

            spinner = Spinner(f"Grid size: {grid_size:>2} | Iterative ")
            spinner.start()
            _, iterative_stats, _ = mdp.to_LMDP_TDR(find_best_lmbda=True, vectorized=False)
            spinner.stop()

            vectorized_times.append(vectorized_stats.get_total_time())
            iterative_times.append(iterative_stats.get_total_time())
            line = f"| {mdp.num_states:<10} | {vectorized_stats.get_total_time():<21.6f} | {iterative_stats.get_total_time():<21.6f} |"
            table_lines.append(line)

    except KeyboardInterrupt:
        # If Ctrl + c is detected, stop possible spinner threads to guarantee a succsessful termination of the program
        print(f"Stopping spinner threads".ljust(40))
        if spinner and spinner.running:
            spinner.stop(interrupted=True)
        exit()
    
    table_lines.append(header_line)
    table_str = "\n".join(table_lines)
    print(table_str)

    with open(save_path, "w") as f:
        f.write(table_str)
    
    
    # Plot the results
    plt.rcParams.update({"text.usetex": True})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
    axes[0].plot(sizes, vectorized_times, label="Vectorized", color="blue", marker="x", linewidth=0.5)
    axes[0].plot(sizes, iterative_times, label="Iterative", color="red", marker="x", linewidth=0.5)
    axes[0].legend()
    axes[0].set_title("Time comparison")
    axes[0].set_ylabel("Time ($s$)")
    
    axes[1].plot(sizes, np.asarray(iterative_times) / np.asarray(vectorized_times), color="black")
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Speedup")
    
    axes[2].set_title("Absolute time difference")
    axes[2].set_ylabel("Time ($s$)")
    axes[2].plot(sizes, np.abs(np.asarray(iterative_times) - np.asarray(vectorized_times)), color="gray")
    
    plt.suptitle("Iterative vs vectorized embedding results")
    fig.supxlabel("Grid size")
    
    plt.savefig(f"{save_path.split('.')[0]}.png", dpi=300, bbox_inches="tight")
    