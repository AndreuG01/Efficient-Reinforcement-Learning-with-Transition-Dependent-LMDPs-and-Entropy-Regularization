from grid_world import GridWorldMDP, GridWorldPlotter
from algorithms import QLearning
import matplotlib.pyplot as plt
import numpy as np


SIMPLE_TEST = [
    "####",
    "#S #",
    "# G#",
    "####"
]


LARGE_TEST = [
    "##########",
    "#    #  G#",
    "#    #   #",
    "#    #   #",
    "#    #   #",
    "#        #",
    "#S       #",
    "##########",
]

WALL_TEST = [
    "##################",
    "#S   ########    #",
    "#    #      #  # #",
    "#    #  #   #  # #",
    "#    #  #   #  # #",
    "#    #  #   #  # #",
    "#       #      #G#",
    "##################"
]

CLIFF = [
    "#################",
    "#               #",
    "# #             #", # TODO: add cliff cells
    "# ############  #",
    "# #   #   #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "# # # # # #  #  #",
    "#S  #   #      G#",
    "#################",
]

if __name__ == "__main__":
    grid_size = 10
    name = f"simple{grid_size}x{grid_size}"
    mdp = GridWorldMDP(
        # map=LARGE_TEST,
        deterministic=True
    )
    
    

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
    # plotter.plot_grid_world(
    #     show_value_function=True,
    #     savefig=False,
    # )
    
    # plotter.plot_stats(savefig=False)
    
    training_epochs = 300000
    epsilon = 0.1
    q_learner = QLearning(mdp, alpha=0.01, gamma=1, epsilon=epsilon, info_every=50000)
    Q, policy, reward = q_learner.train(num_steps=training_epochs)
    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(reward)), reward, label=f"$\epsilon$ = {epsilon}")
    plt.xlabel("Epochs")
    plt.ylabel("Cumulative reward")
    plt.grid()
    plt.legend()
    plt.title("Epsilon effect on large test")
    plt.show()
    
    

    print("Policy")
    print(policy)
    plotter.plot_grid_world(show_value_function=True, savefig=False, policy=policy)