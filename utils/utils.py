import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from custom_palette import CustomPalette
from domains.minigrid_env import MinigridMDP, MinigridLMDP
from domains.grid import MinigridActions
from models.LMDP import LMDP
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP, GridWorldLMDP_TDR
from utils.maps import Maps, Map
import os
from .benchmarks import benchmark_mdp2lmdp_embedding, benchmark_parallel_p
import seaborn as sns
import scipy.stats as stats
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score

def visualize_stochasticity_rewards_embedded_lmdp(state: int, map: Map, num_actions=3, save_fig: bool = True):
    """
    Visualizes the impact of stochasticity on the reward function of a state in an embedded minigrid LMDP.
    
    This function explores how the reward R of a given state changes when the original stochastic MDP is generated 
    with different probabilities of selecting the intended action. The transition probabilities are used to 
    approximate the reward function in an embedded LMDP framework.
    
    Parameters:
    - state (int): The state for which the reward is analyzed.
    - num_actions (int): The number of actions that will be considered
    - map (optional): The layout of the minigrid environment.
    - save_fig (bool, default=True): Whether to save the generated figure or show it.
    
    The function generates three plots:
    1. The entropy term in the Bellman equation for different probabilities of correct action selection.
    2. The computed reward R(s) for the same probabilities.
    3. The controlled transition probabilities for each probability of correct action selection.
    """
    
    palette = CustomPalette()
    
    np.random.seed(31)
    probs = np.arange(0.1, 1, 0.1)
    rand_idx = np.random.randint(0, len(palette))
    color = palette[rand_idx]
    
    plt.rcParams.update({"text.usetex": True})
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    
    ys = []
    rewards = []
    res_probs = []
    
    zero_cols_idx = None
    for main_prob in probs:
        minigrid_mdp = MinigridMDP(
            allowed_actions=MinigridActions.get_actions()[:num_actions],
            map=map,
            behaviour="stochastic",
            stochastic_prob=main_prob
        )
        
        B = minigrid_mdp.P[state, :, :]
        zero_cols = np.all(B == 0, axis=0)
        zero_cols_idx = np.where(zero_cols)[0]
        B = B[:, ~zero_cols]
        B[B == 0] = 1e-10
        B /= np.sum(B, axis=1).reshape(-1, 1)
        log_B = np.where(B != 0, np.log(B), B)
        y = minigrid_mdp.R[state] + np.sum(B * log_B, axis=1)
        B_dagger = np.linalg.pinv(B)
        c = B_dagger @ y
        R = np.log(np.sum(np.exp(c)))
        
        embedded_lmdp = minigrid_mdp.to_LMDP()
        lmdp = LMDP(
            minigrid_mdp.num_states,
            minigrid_mdp.num_terminal_states,
            s0=minigrid_mdp.s0,
            sparse_optimization=False
        )
        lmdp.P = embedded_lmdp.P
        lmdp.R = embedded_lmdp.R
        
        lmdp.compute_value_function()
        next_states = [i for i in range(lmdp.policy.shape[1]) if i not in zero_cols_idx]
        policy = lmdp.policy[state, next_states]
        res_probs.append(policy)
        
        ys.append(y[0])
        rewards.append(R)
        print(lmdp.R)
    
    axes[0].plot(probs, ys, marker="x", color=color)
    axes[1].plot(probs, rewards, color=color, marker="x")
    
    num_colors = max([len(i) for i in res_probs])
    np.random.seed(36)  # Change the seed only because I do not like the colors generated with the previous one.
    colors = [palette[i] for i in np.random.randint(0, len(palette), num_colors)]
    axes[2].grid()
    
    added_labels = set()
    for i, prob_stoch in enumerate(res_probs):
        for j, prob in enumerate(prob_stoch):
            if prob > 0:
                label = f"$s_{{{next_states[j]}}}$" if next_states[j] not in added_labels else None
                axes[2].scatter(probs[i], prob, color=colors[j], label=label, marker="x")
                added_labels.add(next_states[j])
    
    plt.suptitle(f"Effect of Stochasticity on Reward for State {state} in Embedded LMDP ({num_actions} actions)")
    axes[0].set_xlabel("Probability of Correct Action Selection ($a$)\nin the original MDP")
    axes[0].set_ylabel("$\mathbf{y} = \mathcal{R}(s, a) + \sum_{s'}\mathcal{P}(s'\mid s, a) \cdot \log \mathcal{P}(s'\mid s, a)$")
    axes[0].grid()
    axes[1].set_xlabel("Probability of Correct Action Selection ($a$)\nin the original MDP")
    axes[1].set_ylabel("$\mathcal{R}(s)$")
    axes[1].grid()
    axes[2].set_xlabel("Probability of Correct Action Selection ($a$)\nin the original MDP")
    axes[2].set_ylabel("Controlled Transition Probability $a(s'\mid s)$")
    axes[2].legend()
    
    if save_fig:
        plt.savefig("assets/stochasticity_effect_normal_reward_1.png", dpi=300)
    else:
        plt.show()



def compare_value_function_by_stochasticity(map: Map, save_fig: bool = True):
    """
    Compares the value function of an MDP under different stochasticity levels.
    
    This function evaluates how varying the probability of taking the intended transition 
    affects the computed value function in an MDP setting.

    Parameters:
    - map (optional): The environment map, if applicable.
    - grid_size (int, default=3): The size of the grid for the MDP environment.
    """
    palette = CustomPalette()
    map_name = map.name
    
    plt.rcParams.update({"text.usetex": True})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    linewidth = 0.5

    for i, stochasticity in enumerate(np.arange(0.1, 1.0, 0.1)):
        mdp = MinigridMDP(
            map=map,
            allowed_actions=MinigridActions.get_actions(),
            behaviour="deterministic" if stochasticity == 1 else "stochastic",
            stochastic_prob=stochasticity
        )
        mdp.compute_value_function()
        embedded_lmdp = mdp.to_LMDP()
        embedded_lmdp.compute_value_function()
        axes[0].plot([i for i in range(mdp.num_states)], mdp.V, label=f"$p = {round(stochasticity, 1)}$, Iter: ${mdp.stats.iterations}$", linewidth=linewidth, color=palette[i])
        axes[1].plot([i for i in range(embedded_lmdp.num_states)], embedded_lmdp.V, linewidth=linewidth, color=palette[i])
    
    axes[0].set_xlabel("State $s$")
    axes[0].set_ylabel("$V_{MDP}^*(s)$")
    axes[0].legend()
    axes[1].set_xlabel("State $s$")
    axes[1].set_ylabel("$V_{LMDP}^*(s)$")
    
    plt.suptitle(f"Value function comparison of MDP and its embedded LMDP with different stochasticity $p$\nMap: {map_name}")
    
    if save_fig:
        plt.savefig(f"assets/stochasticity_comparison_{map_name}.png", dpi=300)
    else:
        plt.show()


def lmdp_tdr_advantage(save_fig: bool = True):
    """
    A function that has been created to illustrate the advantages of the LMDP with transition-dependent rewards over
    an LMDP with state-dependent rewards.
    """
    map = Maps.CLIFF_WALKING
    mdp = GridWorldMDP(
        map=map,
        behaviour="deterministic"
    )
    lmdp = GridWorldLMDP(
        map=map,
        sparse_optimization=False
    )
    lmdp_tdr = GridWorldLMDP_TDR(
        map=map,
        sparse_optimization=False
    )
    
    # Risky states are those between the index 25 and 34 (both included)
    for i in range(25, 34 + 1):
        mdp.R[i,:] = -20
        mdp.R[i,1] = -5
        lmdp.R[i] = -20
        lmdp_tdr.R[i,:] = -20
        lmdp_tdr.R[i,i+1] = -5
    
    lmdp.compute_value_function()
    mdp.compute_value_function()
    lmdp_tdr.compute_value_function()
    
    output_dir = "LMDP_TDR_advantage"
    
    plotter = GridWorldPlotter(
        mdp,
        name=output_dir
    )
    
    # 1. Save the MAP
    plotter.plot_grid_world(
        show_value_function=False,
        show_prob=False,
        show_actions=False,
        savefig=True,
        save_title="WALKING_CLIFF"
    )
    
    diff_mdp_lmdp = round(np.linalg.norm(mdp.V - lmdp.V), 4)
    diff_mdp_lmdptdr = round(np.linalg.norm(mdp.V - lmdp_tdr.V), 4)
    
    print(f"Squared norm between MDP and LMDP state-dependent: {diff_mdp_lmdp}")
    print(f"Squared norm between MDP and LMDP transition-dependent: {diff_mdp_lmdptdr}")

    palette = CustomPalette()
    plt.rcParams.update({"text.usetex": True})
    
    mdp_color = palette[5]
    lmdp_color = palette[3]
    lmdp_tdr_color = palette[4]

    fig, ax = plt.subplots(figsize=(10, 5))
    states_idx = [i for i in range(len(mdp.V))]
    ax.plot(states_idx, mdp.V, label="MDP", color=mdp_color)
    ax.plot(states_idx, lmdp.V, label="state-dependent LMDP", color=lmdp_color)
    ax.plot(states_idx, lmdp_tdr.V, label="transition-dependent LMDP", color=lmdp_tdr_color)

    rect = patches.Rectangle((24.5, min(mdp.V)), 10, max(mdp.V) - min(mdp.V),
                            linewidth=1.5, edgecolor="red", linestyle="--", facecolor="none", label="Risky states")
    ax.add_patch(rect)

    ax.set_title("Value functions for the Walking Cliff problem")
    ax.set_xlabel("State index")
    ax.set_ylabel("$V(s)$")
    ax.legend()
    ax.grid()
    
    if save_fig:
        plt.savefig(os.path.join(f"assets/{output_dir}", "value_functions.png"), dpi=300, bbox_inches="tight")


    # Calculate R^2 values
    r_squared_lmdp_mdp = r2_score(mdp.V, lmdp.V)
    r_squared_lmdp_tdr_mdp = r2_score(mdp.V, lmdp_tdr.V)

    
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    
    # sns.regplot(x=mdp.V, y=lmdp.V, ax=axes[0], scatter=False, line_kws={"color": "gray", "lw": 1, "linestyle": "--"})
    axes[0].plot(mdp.V, mdp.V, color="gray", linestyle="--", lw=1)
    axes[0].scatter(mdp.V, lmdp.V, color=lmdp_color, zorder=3)
    axes[0].set_title(f"$\mathcal{{M}}$ vs $\mathcal{{L}}$: $R^2 = {r_squared_lmdp_mdp:.3f}$")
    axes[0].set_xlabel("$V_{\mathcal{M}}(s)$")
    axes[0].set_ylabel("$V_{\mathcal{L}}(s)$")
    
    
    # sns.regplot(x=mdp.V, y=lmdp_tdr.V, ax=axes[1], scatter=False, line_kws={"color": "gray", "lw": 1, "linestyle": "--"})
    axes[1].plot(mdp.V, mdp.V, color="gray", linestyle="--", lw=1)
    axes[1].scatter(mdp.V, lmdp_tdr.V, color=lmdp_tdr_color, zorder=3)
    axes[1].set_title(f"$\mathcal{{M}}$ vs $\mathcal{{L'}}$: $R^2 = {r_squared_lmdp_tdr_mdp:.3f}$")
    axes[1].set_xlabel("$V_{\mathcal{M}}(s)$")
    axes[1].set_ylabel("$V_{\mathcal{L\'}}(s)$")
    
    if save_fig:
        plt.savefig(os.path.join(f"assets/{output_dir}", "correlation_plots.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def uniform_assumption_plot(save_fig: bool = True):
    lmdp = GridWorldLMDP(
            map=Map(grid_size=15),
            # allowed_actions=[
            #     MinigridActions.ROTATE_LEFT,
            #     MinigridActions.ROTATE_RIGHT,
            #     MinigridActions.FORWARD,
            # ],
            sparse_optimization=False
        )
    probs = np.arange(0.1, 1.0, 0.1)
    np.random.seed(23)
    state = {}
    for s in range(lmdp.num_non_terminal_states):
        non_zero = np.where(lmdp.P[s] != 0)[0]
        state[s] = np.random.choice(non_zero)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = CustomPalette()
    plt.rcParams.update({"text.usetex": True})
    colors = [palette[i] for i in range(len(probs))]
    
    value_functions = []
    
    for i, stochastic_prob in enumerate(probs):
        for s in range(lmdp.num_non_terminal_states):
            non_zero = np.where(lmdp.P[s] != 0)[0]
            lmdp.P[s, non_zero] = (1 - stochastic_prob) / max(len(non_zero) - 1, 1)
            lmdp.P[s, non_zero[0]] = stochastic_prob

        lmdp.compute_value_function()
        value_functions.append(lmdp.V)
        iters = lmdp.stats.iterations
        axes[0].plot([i for i in range(len(lmdp.V))], lmdp.V, label=f"$p = {round(stochastic_prob, 1)}$. ${iters}$ iters", color=colors[i])
    
    axes[0].set_title("Value functions")
    axes[0].set_xlabel("State index $s$")
    axes[0].set_ylabel(r"$V_{p_i}(s)$")
    axes[0].legend()
    correlation_matrix = np.corrcoef(value_functions)
    

    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", xticklabels=[f"$V_{{p_{i+1}}}(\cdot)$" for i in range(len(value_functions))], yticklabels=[f"$V_{{p_{i+1}}}(\cdot)$" for i in range(len(value_functions))], ax=axes[1])
    axes[1].set_title("Pearson correlation matrix")
    plt.suptitle("Impact of Transition Probability Bias on Value Functions and Convergence in LMDPs", fontsize=14)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig("assets/impact_transition_probability.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def generate_vi_pi_table(save_path: str = "assets/vi_pi_table.txt"):
    grid_sizes = [2, 5, 10, 20, 50, 60, 70, 80, 90, 100]
    table_lines = []
    
    table_lines.append("| Num States | VI Iterations | PI Iterations | VI Time (s) | PI Time (s) |")
    table_lines.append("|------------|---------------|---------------|-------------|-------------|")

    for i, size in enumerate(grid_sizes):
        mdp_stats, lmdp_stats = benchmark_mdp2lmdp_embedding(map=Map(grid_size=size), savefig=False, visual=False)
        
        line = f"| {mdp_stats.num_states:<10} | {mdp_stats.iterations:<13} | {lmdp_stats.iterations:<13} | {mdp_stats.time:<11.4f} | {lmdp_stats.time:<11.4f} |"
        table_lines.append(line)
        
    table_str = "\n".join(table_lines)
    print(table_str)
    
    with open(save_path, "w") as f:
        f.write(table_str)
        

def generate_parallel_p_table(save_path: str = "assets/parallel_p_table.txt"):
    table_lines = []
    
    for behaviour in ["deterministic", "stochastic"]:
        first_line = "|--------------------------------------|"
        table_lines.append(first_line)
        title = f"| {behaviour} MDP"
        title += (" " * (len(first_line) - len(title) - 1)) + "|"
        # title += "|"e
        table_lines.append(title)
        table_lines.append("|--------------------------------------|")
        table_lines.append("| Num States | Cores | Time (s)        |")
        table_lines.append("|------------|-------|-----------------|")
        
        res = benchmark_parallel_p(
            behaviour=behaviour,
            savefig=True,
            visual=True
        )
        
        for core in sorted(res.keys()):
            for state_count, time in res[core]:
                table_lines.append(f"| {state_count:<10} | {core:<5} | {time:<15.4f} |")
        
        table_lines.append(first_line)

    table_str = "\n".join(table_lines)
    print(table_str)
    
    with open(save_path, "w") as f:
        f.write(table_str)

    