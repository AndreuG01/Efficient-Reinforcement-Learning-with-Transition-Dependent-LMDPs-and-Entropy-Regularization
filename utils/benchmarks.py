from domains.grid_world import GridWorldMDP
from domains.minigrid_env import MinigridMDP, MinigridLMDP
from domains.minigrid_env import MinigridMDP, MinigridActions
from utils.maps import Maps
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from joblib import cpu_count
import numpy as np
from custom_palette import CustomPalette



def benchmark_value_iteration(savefig: bool = True):
    time_efficient = []
    time_inefficient = []
    min_size = 2
    max_size = 20
    for size in range(min_size, max_size):
        gridworld_mdp = GridWorldMDP(
            grid_size=size,
            # map=Maps.CLIFF,
            deterministic=True
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



def benchmark_parallel_p(savefig: bool = True):
    min_grid = 10
    max_grid = 65
    limit_core = 21
    results = []
    num_states = []
    for jobs in range(1, min(cpu_count(), limit_core)):
        tmp_results = []
        for grid_size in np.arange(min_grid, max_grid, 1):
            minigrid_lmdp = MinigridLMDP(
                grid_size=grid_size,
                allowed_actions=[
                    MinigridActions.ROTATE_LEFT,
                    MinigridActions.ROTATE_RIGHT,
                    MinigridActions.FORWARD,
                    MinigridActions.PICKUP,
                    MinigridActions.DROP,
                    MinigridActions.TOGGLE,
                ],
                benchmark_p=True,
                threads=jobs,
                sparse_optimization=False
            )
            num_states.append(minigrid_lmdp.num_states)
            tmp_results.append(minigrid_lmdp.p_time)
            print(f"Grid size: {grid_size}. Jobs: {jobs}. Total time: {minigrid_lmdp.p_time:2f} sec")
        
        results.append(tmp_results)
    
    palette = CustomPalette()
    
    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update({
        "text.usetex": True,
    })
    
    for i, job_res in enumerate(results):
        plt.plot([i for i in range(len(job_res))], job_res, label=f"{i + 1} cores", color=palette[i])
    
    plt.title("Parallelization impact on the generation time of transition matrix $\mathcal{P}$.", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid()
    plt.xlabel("Number of states")
    xticks_positions = plt.gca().get_xticks()
    tick_labels = [num_states[int(i)] if i >= 0 else "" for i in xticks_positions]
    tick_labels[0] = ""
    tick_labels[-1] = ""
    plt.xticks(xticks_positions, tick_labels)
    plt.ylabel("Time ($s$)")
    
    if savefig:
        plt.savefig("assets/benchmark/value_iteration.png", dpi=300)
    else:
        plt.show()