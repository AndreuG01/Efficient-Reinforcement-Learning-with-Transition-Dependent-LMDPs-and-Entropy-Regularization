from grid_world import GridWorldMDP, GridWorldPlotter
from algorithms import QLearning
import matplotlib.pyplot as plt
import numpy as np
from maps import Maps



if __name__ == "__main__":
    grid_size = 2
    name = f"cliff"
    mdp = GridWorldMDP(
        grid_size=grid_size,
        map=Maps.CLIFF,
        deterministic=True
    )
    
    # print(mdp.P.shape)

    mdp.compute_value_function()
    
    # # for i in range(len(policy)):
    # #     print(f"State {i}: action {policy[i]}")
    # # print(policy)
    
    # # mdp.print_rewards()
    # # mdp.print_action_values(V)
    
    # # mdp.print_grid()
    
    
    plotter = GridWorldPlotter(mdp,
        figsize=(7, 5),
        name=name,
    )
    plotter.plot_grid_world(
        show_value_function=True,
        savefig=False,
        multiple_actions=True
    )
    
    # plotter.plot_stats(savefig=False)
    
    training_epochs = 4000000
    epsilon = 1
    q_learner = QLearning(
        mdp,
        alpha=0.01,
        gamma=1,
        epsilon=epsilon,
        info_every=50000,
        # epsilon_decay=1.0
    )
    Q, policy, reward = q_learner.train(num_steps=training_epochs, multiple_actions=True)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(reward)), reward, label=f"$\epsilon$ = {epsilon}")
    plt.xlabel("Epochs")
    plt.ylabel("Cumulative reward")
    plt.grid()
    plt.legend()
    plt.title("Epsilon decay effect on walls test", fontsize=14, fontweight="bold")
    # plt.savefig("assets/walls/q_learning_reward.png", dpi=300)
    plt.show()
    
    plotter.plot_grid_world(show_value_function=True, savefig=False, policy=policy)
    
    