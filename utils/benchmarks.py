from domains.grid_world import GridWorldMDP
from domains.minigrid_env import MinigridMDP, MinigridLMDP
from domains.minigrid_env import MinigridMDP
from domains.grid import MinigridActions
from utils.maps import Map
from utils.state import Object
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from joblib import cpu_count
import numpy as np
from custom_palette import CustomPalette
from utils.stats import ModelBasedAlgsStats


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
        
        
def benchmark_lmdp2mdp_embedding(savefig: bool = True, grid_size: int = None, map: list[str] = None, objects: list[Object] = None, name: str = None):
    
    custom_palette = CustomPalette()
    lmdp_color = custom_palette[3]
    mdp_color = custom_palette[4]
    
    save_name = name
    if not grid_size:
        assert map and name, "Must provide a map and its name if no grid size is specified"
    elif not map:
        name = f"Simple grid ${grid_size}\\times{grid_size}$"
        save_name = f"simple_grid_{grid_size}"
    
    
    
    minigrid_lmdp = MinigridLMDP(
        grid_size=grid_size,
        map=map,
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            MinigridActions.PICKUP,
            MinigridActions.DROP,
            MinigridActions.TOGGLE,
        ],
        objects=objects, 
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
    plt.suptitle(f"LMDP and its embedded MDP comparison. {name}", fontsize=14, fontweight="bold")
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
    plt.suptitle(f"LMDP and its embedded MDP comparison. {name}. {minigrid_lmdp.num_states} states", fontsize=14, fontweight="bold")
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
    
    
    cliff_states = [state for state in range(minigrid_mdp.num_states) if minigrid_mdp.minigrid_env.custom_grid.is_cliff(minigrid_mdp.minigrid_env.custom_grid.state_index_mapper[state])]
    
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