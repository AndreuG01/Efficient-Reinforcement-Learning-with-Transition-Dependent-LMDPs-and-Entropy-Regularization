from domains.grid_world import GridWorldMDP, GridWorldPlotter
from domains.minigrid_env import MinigridMDP, MinigridActions
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import numpy as np
from utils.maps import Maps



if __name__ == "__main__":
    grid_size = 4
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
    
    
    
        
    plotter = GridWorldPlotter(
        gridworld_mdp,
        figsize=(7, 5),
        name=name,
    )
    # gridworld_mdp.compute_value_function()
    
    # plotter.plot_grid_world(
    #     show_value_function=True,
    #     savefig=True,
    #     multiple_actions=True,
    #     save_title="gridworld_value_iteration_policy.png"
    # )
    
    # plotter.plot_stats(savefig=False)
    
    explorer = QLearningHyperparameterExplorer(
        minigrid_mdp,
        # alphas=[0.15, 0.1],
        alphas=[0.15, 0.1, 0.05, 0.01, 0.001],
        alphas_decays=[0, 100, 1000, 10000],
        # alphas_decays=[0],
        gammas=[1],
        epochs=2000000,
        out_path="assets/exploration/minigrid_cliff",
        domain_name="Cliff"
    )
    
    explorer.test_hyperparameters()
    
    # training_epochs = 100000
    # hyperparameters = QLearningHyperparameters(
    #     alpha = 0.01,
    #     alpha_decay=0,
    #     gamma=1
    # )
    # epsilon = 1
    # q_learner = QLearning(
    #     minigrid_mdp,
    #     alpha=hyperparameters.alpha,
    #     gamma=hyperparameters.gamma,
    #     epsilon=epsilon,
    #     info_every=100000,
    #     alpha_decay=hyperparameters.alpha_decay
    # )
    # _, policies, reward, errors = q_learner.train(num_steps=training_epochs, multiple_actions=False, multiple_policies=False)
    
    # q_plotter = QLearningPlotter(save_path="assets/minigrid_hyperparameters", domain_name="Simple Grid", vertical=True)
    # q_plotter.plot([reward], [errors], [hyperparameters])
    
    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(len(reward)), reward, label=f"$\epsilon$ = {epsilon}")
    # plt.xlabel("Epochs")
    # plt.ylabel("Cumulative reward")
    # plt.grid()
    # plt.legend()
    # plt.title(f"Epsilon decay effect on {name}", fontsize=14, fontweight="bold")
    # # plt.savefig(f"assets/{name}/minigrid_q-learning_reward.png", dpi=300)
    # # plt.show()
    
    # # plotter.plot_grid_world(show_value_function=True, savefig=False, policy=policy)
    
    
    # # plotter.plot_grid_world(
    # #     savefig=True,
    # #     show_value_function=False,
    # #     save_title="minigrid_q-learning_policy.png",
    # #     policy=policies[0][1]
    # # )
    
    # minigrid_mdp.visualize_policy(
    #     policies=policies,
    #     num_times=1,
    #     save_gif=True,
    #     save_path=f"assets/{name}/minigrid_q-learning_policy.gif"
    # )
    
    