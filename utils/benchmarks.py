from domains.grid_world import GridWorldMDP
from domains.minigrid_env import MinigridMDP
from domains.minigrid_env import MinigridMDP, MinigridActions
from utils.maps import Maps
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator




def benchmark_value_iteration():
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
    plt.show()