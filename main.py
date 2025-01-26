from domains.grid_world import GridWorldMDP, GridWorldPlotter
from domains.minigrid_env import MinigridMDP, MinigridActions
from algorithms import QLearning
import matplotlib.pyplot as plt
import numpy as np
from utils.maps import Maps



if __name__ == "__main__":
    grid_size = 10
    name = f"cliff"
    
    gridworld_mdp = GridWorldMDP(
        grid_size=grid_size,
        map=Maps.CLIFF,
        deterministic=True
    )
    
    minigrid_mdp = MinigridMDP(
        grid_size=grid_size,
        map=Maps.CLIFF,
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD
        ]
    )
    print(f"Gridworld num states: {gridworld_mdp.num_states}")
    print(f"Minigrid num states: {minigrid_mdp.num_states}")

    # minigrid_mdp.visualize_policy(
    #     save_gif=False,
    #     save_path="assets/cliff.gif"
    # )
    

    # gridworld_mdp.compute_value_function()
    # minigrid_mdp.compute_value_function()
    
    # print(gridworld_mdp.stats.iterations)
    # print(minigrid_mdp.stats.iterations)
    
    
    
        
    plotter = GridWorldPlotter(gridworld_mdp,
        figsize=(7, 5),
        name=name,
    )
    gridworld_mdp.compute_value_function()
    
    plotter.plot_grid_world(
        show_value_function=True,
        savefig=False,
        multiple_actions=True
    )
    
    # plotter.plot_stats(savefig=False)
    
    training_epochs = 2000000
    epsilon = 1
    q_learner = QLearning(
        gridworld_mdp,
        alpha=0.01,
        gamma=1,
        epsilon=epsilon,
        info_every=100000,
        # alpha_decay=1000
    )
    Q, policies, reward = q_learner.train(num_steps=training_epochs, multiple_actions=False, multiple_policies=False)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(reward)), reward, label=f"$\epsilon$ = {epsilon}")
    plt.xlabel("Epochs")
    plt.ylabel("Cumulative reward")
    plt.grid()
    plt.legend()
    plt.title("Epsilon decay effect on walls test", fontsize=14, fontweight="bold")
    # plt.savefig("assets/walls/q_learning_reward.png", dpi=300)
    plt.show()
    
    # plotter.plot_grid_world(show_value_function=True, savefig=False, policy=policy)
    
    
    plotter.plot_grid_world(savefig=False, show_value_function=True, policy=policies[0][1])
    
    minigrid_mdp.visualize_policy(
        policies=policies,
        num_times=1,    
    )
    
    