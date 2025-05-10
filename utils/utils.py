import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from custom_palette import CustomPalette
from domains.minigrid_env import MinigridMDP, MinigridLMDP, MinigridLMDP_TDR
from domains.grid import MinigridActions, GridWorldActions
from models.LMDP import LMDP
from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP, GridWorldLMDP_TDR
from utils.maps import Maps, Map
import os
from .benchmarks import benchmark_mdp2lmdp_embedding, benchmark_parallel_p
import seaborn as sns
import scipy.stats as stats
from scipy.sparse import csr_matrix
from sklearn.metrics import r2_score
from typing import Literal
from tqdm import tqdm
import itertools
from .spinner import Spinner

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
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=probs[0], vmax=probs[-1])
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
        axes[0].plot([i for i in range(len(lmdp.V))], lmdp.V, label=f"$p = {round(stochastic_prob, 1)}$. ${iters}$ iters", color=cmap(norm(stochastic_prob)))
    
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
    
    results = []
    
    for behaviour in ["deterministic", "stochastic"]:
        first_line = "|--------------------------------------|"
        table_lines.append(first_line)
        title = f"| {behaviour} MDP"
        title += (" " * (len(first_line) - len(title) - 1)) + "|"
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
        
        results.append(res)

    table_str = "\n".join(table_lines)
    print(table_str)
    
    with open(save_path, "w") as f:
        f.write(table_str)
    
    
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    
    plt.rcParams.update({
        "text.usetex": True,
    })
    palette = CustomPalette()
    
    for i, behaviour in zip(range(len(results)), ["deterministic", "stochastic"]):
        axes[i].grid()
        axes[i].set_title(behaviour.capitalize(), fontsize=10, fontweight="bold")
        for j, core in enumerate(results[i].keys()):
            times = [elem[1] for elem in results[i][core]]
            states = [elem[0] for elem in results[i][core]]
            if j == 0:
                axes[i].plot(states, times, label=f"{core} core", color=palette[j], linestyle="--")
            else:
                axes[i].plot(states, times, label=f"{core} cores", color=palette[j], linewidth=1, marker="x")
        if i == 0:
            axes[i].tick_params(labelbottom=False)
            axes[i].set_xlabel("")
            axes[i].legend(loc="upper left")

    
    fig.supxlabel(r"Number of states")
    fig.supylabel(r"Time ($s$)")
    fig.suptitle("Parallelization impact on the generation time of transition matrix $\mathcal{P}$.", fontsize=14, fontweight="bold")

    plt.tight_layout()
    

    plt.savefig(f"assets/benchmark/parallel_p_combined.png", dpi=300, bbox_inches="tight")


def different_gammas_plot(save_fig: bool = True):
    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.tight_layout()
    gammas = np.arange(0, 1.1, 0.1)
    
    cmap = plt.colormaps["turbo"]
    norm = Normalize(vmin=gammas.min(), vmax=gammas.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    np.random.seed(43)
    r = np.random.randint(-10, 0, size=(21, 4))
    r[20] = np.array([0, 0, 0, 0])


    for i, gamma in enumerate(gammas):
        mdp = GridWorldMDP(
            map=Maps.GRIDWORLD_MDP_MYOPIC,
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="deterministic",
            gamma=gamma,
        )
        
        if i == 1:
            for s in range(mdp.num_states):
                plt.scatter(s, np.max(mdp.R[s]), color="#FF00C6", marker="x", zorder=3, s=50, label="$\max_{a}\mathcal{R}(s,a)$" if s == 0 else None)
    
        mdp.compute_value_function()

        plt.plot([i for i in range(len(mdp.V))], mdp.V, color=cmap(norm(gamma)))
    
    ax.set_xlabel("State $s$")
    ax.set_ylabel("$V(s)$")
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("$\gamma$")
    ax.set_title("Value function for different discount factor ($\gamma$) values.")
    plt.legend()
    
    if save_fig:
        plt.savefig("assets/different_gamma.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def different_temperature_plots(model_type: Literal["MDP", "LMDP"] = "MDP", save_fig: bool = True):
    assert model_type in ["MDP", "LMDP"], f"Invalid model type. Only valids: {['MDP', 'LMDP']}"
    
    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.tight_layout()
    temperatures = np.arange(0.1, 10.1, 0.1)
    
    cmap = plt.colormaps["turbo"]
    norm = Normalize(vmin=temperatures.min(), vmax=temperatures.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    map = Map(grid_size=5)

    for i, temp in enumerate(temperatures):
        if model_type == "MDP":
            model = GridWorldMDP(
                map=map,
                allowed_actions=GridWorldActions.get_actions()[:4],
                behaviour="deterministic",
                temperature=temp
            )
        else:
            model = GridWorldLMDP(
                map=map,
                allowed_actions=GridWorldActions.get_actions()[:4],
                lmbda=temp
            )
    
        model.compute_value_function()
        plt.plot([i for i in range(len(model.V))], model.V, color=cmap(norm(temp)))
    
    temp_name = "beta" if model_type == "MDP" else "lambda"
    
    ax.set_xlabel("State $s$")
    ax.set_ylabel("$V(s)$")
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"$\{temp_name}$")
    ax.set_title(f"Value function for different temperature parameters ($\{temp_name}$) values.")
    plt.grid()
    if save_fig:
        plt.savefig(f"assets/{model_type}_different_temperature.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def regularized_embedding_error_plot(map: Map, min_temp: float = 0.1, max_temp: float = 3, temp_step: float = 0.1, save_fig: bool = True):

    temperatures = np.arange(min_temp, max_temp, temp_step)
    errors_state_dependent = []
    errors_trans_dependent = []
    
    for temp in temperatures:
        mdp = MinigridMDP(
            map=map,
            behaviour="deterministic",
            allowed_actions=MinigridActions.get_actions(),
            temperature=temp
        )
        
        embedded_lmdp_state = mdp.to_LMDP(lmbda=temp)
        embedded_lmdp_state.compute_value_function()
        embedded_lmdp_trans, _ = mdp.to_LMDP_TDR(temp)
        embedded_lmdp_trans.compute_value_function()
        mdp.compute_value_function()
        
        error_state = np.mean(np.square(mdp.V - embedded_lmdp_state.V))
        error_trans = np.mean(np.square(mdp.V - embedded_lmdp_trans.V))
        errors_state_dependent.append(error_state)
        errors_trans_dependent.append(error_trans)
    
    palette = CustomPalette()
    plt.rcParams.update({"text.usetex": True})
    fig = plt.figure(figsize=(8, 5))
    plt.tight_layout()
    
    plt.title(f"MDP-LMDP embedding error comparison on LMDP with state and transition-dependent rewards.\nMap: {map.name}")
    
    min_state_dep = min(errors_state_dependent)
    min_trans_dep = min(errors_trans_dependent)
    
    plt.plot(temperatures, errors_state_dependent, label="State-dependent", color=palette[2])
    plt.plot(temperatures, errors_trans_dependent, label="Transition-dependent", color=palette[17])
    print([val for val in errors_state_dependent if val == min_state_dep])
    plt.scatter([temperatures[i] for i, val in enumerate(errors_state_dependent) if val == min_state_dep][0], min_state_dep, color=palette[3], zorder=3, marker="x", s=50, label="Min value")
    plt.grid()
    
    
    plt.ylabel(r"MSE$(V_\mathcal{M}, V_\mathcal{L})$")
    plt.xlabel("Regularization parameter value")
    
    plt.legend()
    
    if save_fig:
        save_map_name = map.name.lower().replace(" ", "_")
        plt.savefig(f"assets/reg_no_reg_embedding_error_{save_map_name}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def create_models(map, f, t, temp_mdp, temp_lmdp, behaviour, stochastic_prob):
    if f == "MDP":
        mdp = GridWorldMDP(
            map=map,
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour=behaviour,
            stochastic_prob=stochastic_prob,
            temperature=temp_mdp
        )
        lmdp, _ = mdp.to_LMDP_TDR(lmbda=temp_lmdp)
        lmdp = GridWorldLMDP_TDR(
            map=map,
            allowed_actions=GridWorldActions.get_actions()[:4],
            lmdp_tdr=lmdp
        )
    else:
        lmdp = GridWorldLMDP(
            map=map,
            allowed_actions=GridWorldActions.get_actions()[:4],
        )
        mdp = lmdp.to_MDP()
        mdp = GridWorldMDP(
            map=map,
            allowed_actions=GridWorldActions.get_actions()[:4],
            mdp=mdp
        )
    mdp.compute_value_function()
    lmdp.compute_value_function()
    return mdp, lmdp


def compute_value_metrics(mdp, lmdp):
    mse = np.mean(np.square(mdp.V - lmdp.V))
    r2 = r2_score(mdp.V, lmdp.V)
    corr = np.corrcoef(mdp.V, lmdp.V)[0, 1]
    return mse, r2, corr


def embedding_value_function_reg(map, f="MDP", t="LMDP", behaviour="deterministic", stochastic_prob=0.9, save_fig=False):
    assert f in ["MDP", "LMDP"]
    assert t in ["MDP", "LMDP"]
    assert t != f
    assert behaviour in ["deterministic", "stochastic", "mixed"]

    temps = np.arange(1, 7, 0.05)
    lmdp_color = plt.get_cmap("Reds")
    mdp_color = plt.get_cmap("Blues")
    normalizer = Normalize(vmin=temps[0], vmax=temps[-1])
    sm_lmbda = cm.ScalarMappable(cmap=lmdp_color, norm=normalizer)
    sm_beta = cm.ScalarMappable(cmap=mdp_color, norm=normalizer)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    ax_val = fig.add_subplot(gs[0, :])
    ax_mse = fig.add_subplot(gs[1, 0])
    ax_r2 = fig.add_subplot(gs[1, 1])
    ax_corr = fig.add_subplot(gs[1, 2])

    mse_vals, r2_vals, corr_vals = [], [], []

    for temp in temps:
        mdp, lmdp = create_models(map, f, t, temp, temp, behaviour, stochastic_prob)
        mse, r2, corr = compute_value_metrics(mdp, lmdp)

        mse_vals.append(mse)
        r2_vals.append(r2)
        corr_vals.append(corr)

        ax_val.plot(range(len(mdp.V)), mdp.V, color=mdp_color(normalizer(temp)))
        ax_val.plot(range(len(lmdp.V)), lmdp.V, color=lmdp_color(normalizer(temp)))

    ax_val.set_xlabel(r"State $s$")
    ax_val.set_ylabel(r"$V(s)$")
    fig.colorbar(sm_beta, ax=ax_val).set_label(r"MDP temperature ($\beta$)")
    fig.colorbar(sm_lmbda, ax=ax_val).set_label(r"LMDP temperature ($\lambda$)")

    optimal_color = "#FF00C6"

    def plot_metric(ax, data, label, ylabel):
        ax.plot(temps, data, color="#00CC15")
        idx = np.argmax(data) if "Corr" in ylabel or "R^2" in ylabel else np.argmin(data)
        ax.scatter(temps[idx], data[idx], color=optimal_color, marker="x", zorder=3, label=f"{label} $\lambda = {round(temps[idx], 2)}$")
        ax.set_title(ylabel)
        ax.set_xlabel(r"Temperature $\lambda = \beta$")
        ax.legend()
        ax.grid()

    plot_metric(ax_mse, mse_vals, "Min", "MSE")
    plot_metric(ax_r2, r2_vals, "Max", r"$R^2$")
    plot_metric(ax_corr, corr_vals, "Max", r"Correlation ($\rho$)")

    plt.suptitle(f"{f} to {t} embedding. Stochastic MDP with prob ${stochastic_prob if behaviour != 'deterministic' else '1'}$. Map: {map.name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fig:
        name = map.name.lower().replace(" ", "_")
        path = f"assets/{f}_to_{t}_prob_{stochastic_prob if behaviour != 'deterministic' else 'det'}_{name}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def embedding_errors_different_temp(map, f="MDP", t="LMDP", behaviour="deterministic", stochastic_prob=0.9, save_fig=False):
    assert f in ["MDP", "LMDP"]
    assert t in ["MDP", "LMDP"]
    assert t != f
    assert behaviour in ["deterministic", "stochastic", "mixed"]

    mdp_temps = [1, 2, 3, 4]
    lmdp_range = np.arange(1, 8, 0.2)

    plt.rcParams.update({"text.usetex": True})
    fig, axes = plt.subplots(1, 3, figsize=(25, 5))
    fig.supxlabel(r"Temperature $\lambda$")

    for beta in mdp_temps:
        mse_list, r2_list, corr_list = [], [], []
        for lmbda in lmdp_range:
            mdp, lmdp = create_models(map, f, t, beta, lmbda, behaviour, stochastic_prob)
            mse, r2, corr = compute_value_metrics(mdp, lmdp)
            mse_list.append(mse)
            r2_list.append(r2)
            corr_list.append(corr)

        def plot(ax, data, label, metric):
            ax.plot(lmdp_range, data, label=label)
            idx = np.argmax(data) if metric != "MSE" else np.argmin(data)
            ax.scatter(lmdp_range[idx], data[idx], color="red", marker="x", zorder=3)

        plot(axes[0], mse_list, f"MDP: $\\beta = {beta}$", "MSE")
        plot(axes[1], r2_list, f"MDP: $\\beta = {beta}$", "R2")
        plot(axes[2], corr_list, f"MDP: $\\beta = {beta}$", "Corr")

    for ax, title, ylabel in zip(axes, ["MSE", r"$R^2$", "Correlation"], ["MSE", r"$R^2$", r"$\\rho$"]):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.suptitle(f"{f} to {t} embedding. Stochastic MDP with prob ${stochastic_prob if behaviour != 'deterministic' else '1'}$. Map: {map.name}")
    if save_fig:
        name = map.name.lower().replace(" ", "_")
        path = f"assets/{f}_to_{t}_prob_{stochastic_prob if behaviour != 'deterministic' else 'det'}_{name}_lmbda_choosing.png"
        # plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.savefig("assets/new_version_2.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def policies_comparison(
    mdp: GridWorldMDP | MinigridMDP,
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

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_big = fig.add_subplot(gs[1, :])

    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.1, top=0.85, wspace=0.3, hspace=0.4)

    if not hasattr(mdp, "policy"):
        mdp.compute_value_function()

    lmdp_1, _ = mdp.to_LMDP_TDR(find_best_lmbda=False)
    lmdp_2, _ = mdp.to_LMDP_TDR(find_best_lmbda=True)

    game_times = 1000

    mdp_policy = mdp.to_LMDP_policy().astype(np.float64)
    stats_mdp = mdp.visualize_policy(num_times=game_times, show_window=False)
    print(f"MDP STATS: {stats_mdp.GAME_INFO}")
    
    if type(mdp) == GridWorldMDP:
        new_ldmp_1 = GridWorldLMDP_TDR(
            map=mdp.environment.custom_grid.map,
            allowed_actions=mdp.allowed_actions,
            lmdp_tdr=lmdp_1,
            verbose=False
        )
        
        new_ldmp_2 = GridWorldLMDP_TDR(
            map=mdp.environment.custom_grid.map,
            allowed_actions=mdp.allowed_actions,
            lmdp_tdr=lmdp_2,
            verbose=False
        )
    elif type(mdp) == MinigridMDP:
        new_ldmp_1 = MinigridLMDP_TDR(
            map=mdp.environment.custom_grid.map,
            allowed_actions=mdp.allowed_actions,
            lmdp=lmdp_1,
            verbose=False
        )
        new_ldmp_2 = MinigridLMDP_TDR(
            map=mdp.environment.custom_grid.map,
            allowed_actions=mdp.allowed_actions,
            lmdp=lmdp_2,
            verbose=False
        )
        

    new_ldmp_1.compute_value_function()
    kl_1 = kl_divergence(mdp_policy, new_ldmp_1.policy)
    print(f"kl_1: {kl_1}, mse1: {np.mean(np.square(mdp_policy - new_ldmp_1.policy))}")
    policy_1 = new_ldmp_1.policy
    stats_lmdp_1 = new_ldmp_1.visualize_policy(policies=[(0, policy_1)], num_times=game_times, show_window=False)
    print(f"LMDP STATS 1: {stats_lmdp_1.GAME_INFO}")

    new_ldmp_2.compute_value_function()
    kl_2 = kl_divergence(mdp_policy, new_ldmp_2.policy)
    print(f"kl_2: {kl_2}, mse2: {np.mean(np.square(mdp_policy - new_ldmp_2.policy))}")
    policy_2 = new_ldmp_2.policy
    stats_lmdp_2 = new_ldmp_2.visualize_policy(policies=[(0, policy_2)], num_times=game_times, show_window=False)
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
    ax1.set_title(f"$\pi_{{\mathcal{{M}}}}^{{\\beta = {mdp.temperature}}}(s'\mid s)$")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect("equal")

    im2 = ax2.imshow(policy_data[1], vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title(f"$\pi_{{\mathcal{{L}}}}^{{\lambda = {new_ldmp_1.lmbda}}}(s'\mid s)$.")
    kl_1_formatting = f"\\textbf{{{round(kl_1, 3)}}}" if kl_1 == min_kl else f"{round(kl_1, 3)}"
    ax2.set_xlabel(f"KL$\left(\pi_{{\mathcal{{M}}}} (\cdot\mid s) || \pi_{{\mathcal{{L}}}}(\cdot \mid s)\\right) = {kl_1_formatting}$")
    # ax2.set_xlabel(f"KL$\left(\pi_{{\mathcal{{M}}}} (\cdot\mid s) || \pi_{{\mathcal{{L}}}}(\cdot \mid s)\\right) = {kl_1_formatting}$\nMSE$(V_{{\mathcal{{M}}}}, V_{{\mathcal{{L}}}}) = {round(np.mean(np.square(mdp.V - new_ldmp_1.V)), 2)}$")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect("equal")

    im3 = ax3.imshow(policy_data[2], vmin=vmin, vmax=vmax, cmap=cmap)
    ax3.set_title(f"$\pi_{{\mathcal{{L}}}}^{{\lambda = {round(new_ldmp_2.lmbda, 3)}}}(s'\mid s)$.")
    kl_2_formatting = f"\\textbf{{{round(kl_2, 3)}}}" if kl_2 == min_kl else f"{round(kl_2, 3)}"
    ax3.set_xlabel(f"KL$\left(\pi_{{\mathcal{{M}}}} (\cdot\mid s) || \pi_{{\mathcal{{L}}}}(\cdot \mid s)\\right) = {kl_2_formatting}$")
    # ax3.set_xlabel(f"KL$\left(\pi_{{\mathcal{{M}}}} (\cdot\mid s) || \pi_{{\mathcal{{L}}}}(\cdot \mid s)\\right) = {kl_2_formatting}$.\nMSE$(V_{{\mathcal{{M}}}}, V_{{\mathcal{{L}}}}) = {round(np.mean(np.square(mdp.V - new_ldmp_2.V)), 2)}$")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect("equal")

    
    # Bottom plot
    ax_big.plot(stats_mdp.get_proportion_correct_moves_round(), label="MDP $\mathcal{M}$", linewidth=0.5, color="blue", alpha=0.5)
    ax_big.hlines(np.mean(stats_mdp.get_proportion_correct_moves_round()), xmin=0, xmax=game_times, color="blue", linestyles="--", zorder=3)
    ax_big.plot(stats_lmdp_1.get_proportion_correct_moves_round(), label=f"LMDP $\lambda = {new_ldmp_1.lmbda}$", linewidth=0.5, color="red", alpha=0.5)
    ax_big.hlines(np.mean(stats_lmdp_1.get_proportion_correct_moves_round()), xmin=0, xmax=game_times, color="red", linestyles="--", zorder=3)
    ax_big.plot(stats_lmdp_2.get_proportion_correct_moves_round(), label=f"LMDP $\lambda = {round(new_ldmp_2.lmbda, 3)}$", linewidth=0.5, color="green", alpha=0.5)
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

    plt.suptitle(f"{'Zoomed policy' if zoom else 'Policy'} comparison between MDP $\mathcal{{M}}$ and embedded LMDP $\mathcal{{L}}$ with different $\\beta = \lambda$. Stochastic prob: {mdp.stochastic_prob}. {mdp.environment.custom_grid.map.name}")

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_fig:
        save_map_name = mdp.environment.custom_grid.map.name.lower().replace(" ", "_")
        plt.savefig(f"assets/{'zoomed_' if zoom else ''}policy_comparison_{save_map_name}_p_{mdp.stochastic_prob}_beta_{mdp.temperature}_lmbda_best_{round(lmdp_2.lmbda, 3)}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
        
        

def benchmark_iterative_vectorized_embedding(max_grid_size: int = 60, save_path="assets/iterative_vs_vectorized_embedding.txt"):
    
    table_lines = []
    header_line = "+------------+-----------------------+-----------------------+"
    table_lines.append(header_line)
    table_lines.append("| Num States | Vectorized Time (s)  | Iterative Time (s)    |")
    table_lines.append(header_line)
    
    sizes = np.arange(3, max_grid_size, 2)
    spinner = None
    try:
        for grid_size in sizes:
            mdp = MinigridMDP(
                map=Map(grid_size=grid_size),
                allowed_actions=MinigridActions.get_actions(),
                behaviour="stochastic",
                stochastic_prob=0.3,
                temperature=4.5,
                verbose=False
            )
            spinner = Spinner(f"Grid size: {grid_size:>2} | Vectorized")
            spinner.start()
            _, vectorized_stats = mdp.to_LMDP_TDR(find_best_lmbda=True, vectorized=True)
            spinner.stop()

            spinner = Spinner(f"Grid size: {grid_size:>2} | Iterative ")
            spinner.start()
            _, iterative_stats = mdp.to_LMDP_TDR(find_best_lmbda=True, vectorized=False)
            spinner.stop()

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