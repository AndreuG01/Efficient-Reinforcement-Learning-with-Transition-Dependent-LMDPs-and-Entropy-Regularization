from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP
from domains.minigrid_env import MinigridMDP, MinigridActions, MinigridLMDP
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import numpy as np
from utils.maps import Maps
from utils.benchmarks import benchmark_value_iteration
from minigrid.manual_control import ManualControl

if __name__ == "__main__":
    grid_size = 2
    # map = Maps.WALL_TEST
    # map = None
    # gridworld_lmdp = GridWorldLMDP(
    #     grid_size=grid_size,
    #     map=map
    # )
    
    # gridworld_lmdp.compute_value_function()
    # lmdp_plotter = GridWorldPlotter(
    #     gridworld_lmdp,
    #     figsize=(7, 5),
    # )
    
    # lmdp_plotter.plot_grid_world(
    #     savefig=True,
    #     save_title="LMDP Gridworld Policy",
    #     show_value_function=True,
    #     multiple_actions=False,
    # )
    
    
    # embedded_mdp = GridWorldMDP(
    #     grid_size=grid_size,
    #     mdp=gridworld_lmdp.to_MDP(),
    #     deterministic=False,
    #     map=map
    # )
    
    # embedded_mdp.compute_value_function()
    
    # plotter = GridWorldPlotter(
    #     embedded_mdp,
    #     figsize=(7, 5),
    # )
    
    # plotter.plot_grid_world(
    #     savefig=False,
    #     save_title="Embedded MDP Gridworld Policy Probs colors",
    #     show_value_function=True,
    #     multiple_actions=False,
    #     show_prob=True,
    #     prob_size=3.5,
    #     color_probs=True
    # )

    
    minigrid_mdp = MinigridMDP(
        map=Maps.DOUBLE_DOOR,
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            MinigridActions.PICKUP,
            MinigridActions.DROP,
            MinigridActions.TOGGLE,
        ],
        objects=Maps.DOUBLE_DOOR_OBJECTS,
    )
    
    # print(np.where(minigrid_mdp.P[4, :, :] != 0))
    # for i in range(minigrid_mdp.P.shape[0]):
    #     print(i)
    #     print(np.where(minigrid_mdp.P[i, :, :] != 0))
    # print(minigrid_mdp.num_non_terminal_states)
    # print(np.where(minigrid_mdp.R[:minigrid_mdp.num_non_terminal_states, :] >= 0))
    # print(np.where(minigrid_mdp.R[minigrid_mdp.num_non_terminal_states:, :] < 0))
    # for i in range(minigrid_mdp.P.shape[0]):
    #     print(np.sum(minigrid_mdp.P[i, :, :], axis=1))
    minigrid_mdp.visualize_policy(save_gif=True, save_path="assets/many_states.gif", num_times=1)
    # manual_control = ManualControl(minigrid_mdp.minigrid_env, seed=42)
    # manual_control.start()

    
    

    # minigrid_lmdp = MinigridLMDP(
    #     grid_size=grid_size,
    #     map=Maps.CHALLENGE_DOOR,
    #     allowed_actions=[
    #         MinigridActions.ROTATE_LEFT,
    #         MinigridActions.ROTATE_RIGHT,
    #         MinigridActions.FORWARD,
    #         MinigridActions.PICKUP,
    #         MinigridActions.TOGGLE,
    #         # MinigridActions.DROP,
    #         # MinigridActions.DONE
    #     ],
    #     properties={"orientation": [i for i in range(4)], "blue_door": [False, True], "blue_key": [False, True]}
    # )
    # embedded_mdp = minigrid_lmdp.to_MDP()
    # print("EMBEDDED NUM ACTIONS", embedded_mdp.num_actions)
    
    # minigrid_mdp = MinigridMDP(
    #     grid_size=grid_size,
    #     map=Maps.CHALLENGE_DOOR,
    #     allowed_actions=[
    #         i for i in range(embedded_mdp.num_actions)
    #         # MinigridActions.DONE
    #     ],
    #     properties={"orientation": [i for i in range(4)], "blue_door": [False, True], "blue_key": [False, True]},
    #     mdp=embedded_mdp,
    #     deterministic=False
    # )
    # minigrid_mdp.visualize_policy(
    #     save_gif=True,
    #     save_path="assets/embedded_mdp.gif"
    # )
    
    # minigrid_lmdp.to_MDP()
    
    # minigrid_lmdp.visualize_policy(
    #     save_gif=True,
    #     save_path="assets/minigrid_lmdp.gif"
    # )
    
    # training_epochs = 2000000
    # hyperparameters = QLearningHyperparameters(
    #     alpha = 0.15,
    #     alpha_decay=0,
    #     gamma=1
    # )
    # epsilon = 0.7
    # q_learner = QLearning(
    #     minigrid_mdp,
    #     alpha=hyperparameters.alpha,
    #     gamma=hyperparameters.gamma,
    #     epsilon=epsilon,
    #     info_every=100000,
    #     epsilon_decay=0.999,
    #     alpha_decay=hyperparameters.alpha_decay
    # )
    # _, policies, reward, errors = q_learner.train(num_steps=training_epochs, multiple_actions=False, multiple_policies=False)
    
        
    # minigrid_mdp.visualize_policy(
    #     policies=policies,
    #     save_gif=True,
    #     num_times=10,
    #     save_path="assets/door_gif_last.gif"
    
    # )
    # manual_control = ManualControl(minigrid_mdp.minigrid_env, seed=42)
    # manual_control.start()
    # grid_size = 4
    # name = f"cliff"
    
    # benchmark_value_iteration()
    
    
    
    # print(gridworld_mdp.policy)
    
    # minigrid_mdp = MinigridMDP(
    #     grid_size=grid_size,
    #     map=Maps.CLIFF,
    #     allowed_actions=[
    #         MinigridActions.ROTATE_LEFT,
    #         MinigridActions.ROTATE_RIGHT,
    #         MinigridActions.FORWARD
    #     ]
    # )
    # print(f"Gridworld num states: {gridworld_mdp.num_states}")
    # print(f"Minigrid num states: {minigrid_mdp.num_states}")

    # # minigrid_mdp.visualize_policy(
    # #     save_gif=False,
    # #     save_path="assets/cliff.gif"
    # # )
    

    # # gridworld_mdp.compute_value_function()
    # # minigrid_mdp.compute_value_function()
    
    # # print(gridworld_mdp.stats.iterations)
    # # print(minigrid_mdp.stats.iterations)
    
    
    
        
    
    
    # # plotter.plot_stats(savefig=False)
    
    # explorer = QLearningHyperparameterExplorer(
    #     minigrid_mdp,
    #     # alphas=[0.15, 0.1],
    #     alphas=[0.15, 0.1, 0.05, 0.01, 0.001],
    #     alphas_decays=[0, 100, 1000, 10000],
    #     # alphas_decays=[0],
    #     gammas=[1],
    #     epochs=2000000,
    #     out_path="assets/exploration/minigrid_cliff",
    #     domain_name="Cliff"
    # )
    
    # explorer.test_hyperparameters()
    
    # # training_epochs = 100000
    # # hyperparameters = QLearningHyperparameters(
    # #     alpha = 0.01,
    # #     alpha_decay=0,
    # #     gamma=1
    # # )
    # # epsilon = 1
    # # q_learner = QLearning(
    # #     minigrid_mdp,
    # #     alpha=hyperparameters.alpha,
    # #     gamma=hyperparameters.gamma,
    # #     epsilon=epsilon,
    # #     info_every=100000,
    # #     alpha_decay=hyperparameters.alpha_decay
    # # )
    # # _, policies, reward, errors = q_learner.train(num_steps=training_epochs, multiple_actions=False, multiple_policies=False)
    
    # # q_plotter = QLearningPlotter(save_path="assets/minigrid_hyperparameters", domain_name="Simple Grid", vertical=True)
    # # q_plotter.plot([reward], [errors], [hyperparameters])
    
    # # fig = plt.figure(figsize=(10, 5))
    # # plt.plot(np.arange(len(reward)), reward, label=f"$\epsilon$ = {epsilon}")
    # # plt.xlabel("Epochs")
    # # plt.ylabel("Cumulative reward")
    # # plt.grid()
    # # plt.legend()
    # # plt.title(f"Epsilon decay effect on {name}", fontsize=14, fontweight="bold")
    # # # plt.savefig(f"assets/{name}/minigrid_q-learning_reward.png", dpi=300)
    # # # plt.show()
    
    # # # plotter.plot_grid_world(show_value_function=True, savefig=False, policy=policy)
    
    
    # # # plotter.plot_grid_world(
    # # #     savefig=True,
    # # #     show_value_function=False,
    # # #     save_title="minigrid_q-learning_policy.png",
    # # #     policy=policies[0][1]
    # # # )
    
    # # minigrid_mdp.visualize_policy(
    # #     policies=policies,
    # #     num_times=1,
    # #     save_gif=True,
    # #     save_path=f"assets/{name}/minigrid_q-learning_policy.gif"
    # # )
    
    