import matplotlib.pyplot as plt
import numpy as np
from custom_palette import CustomPalette
from domains.minigrid_env import MinigridMDP, MinigridLMDP
from domains.grid import MinigridActions
from models.LMDP import LMDP

def visualize_stochasticity_rewards_embedded_lmdp(state: int, num_actions=3, map=None, objects=None, grid_size: int = 3, save_fig: bool = True):
    """
    Visualizes the impact of stochasticity on the reward function of a state in an embedded minigrid LMDP.
    
    This function explores how the reward R of a given state changes when the original stochastic MDP is generated 
    with different probabilities of selecting the intended action. The transition probabilities are used to 
    approximate the reward function in an embedded LMDP framework.
    
    Parameters:
    - state (int): The state for which the reward is analyzed.
    - num_actions (int): The number of actions that will be considered
    - map (optional): The layout of the minigrid environment.
    - objects (optional): Objects present in the minigrid.
    - grid_size (int, default=3): The size of the minigrid.
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
            grid_size=grid_size,
            allowed_actions=MinigridActions.get_actions()[:num_actions],
            map=map,
            objects=objects,
            deterministic=False,
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



def compare_value_function_by_stochasticity(map=None, objects=None, grid_size: int = 3, map_name: str = None, save_fig: bool = True):
    """
    Compares the value function of an MDP under different stochasticity levels.
    
    This function evaluates how varying the probability of taking the intended transition 
    affects the computed value function in an MDP setting.

    Parameters:
    - map (optional): The environment map, if applicable.
    - objects (optional): A list of objects or entities within the environment.
    - grid_size (int, default=3): The size of the grid for the MDP environment.
    """
    palette = CustomPalette()
    if map_name is not None:
        assert map_name is not None, "Must provide a name for the map"
    
    plt.rcParams.update({"text.usetex": True})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    linewidth = 0.5

    for i, stochasticity in enumerate(np.arange(0.1, 1.0, 0.1)):
        mdp = MinigridMDP(
            grid_size=grid_size,
            allowed_actions=[
                MinigridActions.ROTATE_LEFT,
                MinigridActions.ROTATE_RIGHT,
                MinigridActions.FORWARD,
                MinigridActions.PICKUP,
                MinigridActions.DROP,
                MinigridActions.TOGGLE
            ],
            map=map,
            objects=objects,
            deterministic=True if stochasticity == 1 else False,
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
