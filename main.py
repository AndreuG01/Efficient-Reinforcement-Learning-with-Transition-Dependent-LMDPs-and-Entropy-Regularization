from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP
from domains.minigrid_env import MinigridMDP, MinigridActions, MinigridLMDP, MinigridLMDP_TDR
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import numpy as np
from utils.maps import Maps
from utils.benchmarks import benchmark_value_iteration, benchmark_parallel_p, benchmark_lmdp2mdp_embedding, benchmark_mdp2lmdp_embedding
from minigrid.manual_control import ManualControl
from custom_palette import CustomPalette
import pickle as pkl
from utils.utils import visualize_stochasticity_rewards_embedded_lmdp

if __name__ == "__main__":
    # benchmark_parallel_p()
    grid_size = 4
    minigrid_mdp = MinigridMDP(
        grid_size=grid_size,
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            # MinigridActions.PICKUP,
            # MinigridActions.DROP,
            # MinigridActions.TOGGLE,
        ],
        map=Maps.CLIFF,
        # objects=Maps.CHALLENGE_DOOR_OBJECTS,
        deterministic=False,
    )
    # minigrid_mdp.visualize_policy(num_times=1)
    # exit()
    # for state in range(minigrid_mdp.num_states):
    #     if minigrid_mdp.minigrid_env.custom_grid.is_cliff(minigrid_mdp.minigrid_env.custom_grid.state_index_mapper[state]):
    #         print(state)
    # print(minigrid_mdp.P[2, :, :])
    # # minigrid_mdp.visualize_policy()
    print(minigrid_mdp.P[100, :, :])
    # exit()
    # minigrid_mdp.compute_value_function()
    embedded_lmdp = minigrid_mdp.to_LMDP()
    
    # print(embedded_lmdp.P[4, :])
    
    stochastic = "0_9"
    
    # # gridworld_mdp.visualize_policy(num_times=1, save_gif=False, save_path=f"assets/cliff_mdp_stochastic_{stochastic}.gif")

    # print(gridworld_mdp.P)
    # embedded_lmdp = gridworld_mdp.to_LMDP()
    # # print(f"Original R: {gridworld_mdp.R}")
    # # print(f"Embedded R: {embedded_lmdp.R}")
    # # print(embedded_lmdp.P)
    minigrid_lmdp = MinigridLMDP(
        grid_size=grid_size,
        allowed_actions=[
            MinigridActions.ROTATE_LEFT,
            MinigridActions.ROTATE_RIGHT,
            MinigridActions.FORWARD,
            # MinigridActions.PICKUP,
            # MinigridActions.DROP,
            # MinigridActions.TOGGLE,
        ],
        map=Maps.CLIFF,
        # objects=Maps.CHALLENGE_DOOR_OBJECTS,
        threads=4,
        sparse_optimization=True,
        lmdp=embedded_lmdp
    )
    minigrid_lmdp.visualize_policy(num_times=1)
    
    print("MDP")
    print(minigrid_mdp.V)
    print("LMDP")
    minigrid_lmdp.compute_value_function()
    print(minigrid_lmdp.V)
    states_to_goal_ideal = minigrid_lmdp.states_to_goal()
    print("mdp", minigrid_mdp.states_to_goal(stochastic=True))
    print("lmdp", minigrid_lmdp.states_to_goal(stochastic=True))
    
    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(len(minigrid_mdp.V)), minigrid_mdp.V, label="MDP")
    # plt.plot(np.arange(len(minigrid_lmdp.V)), minigrid_lmdp.V, label="LMDP")
    # plt.scatter(minigrid_mdp.states_to_goal(), minigrid_mdp.V[minigrid_mdp.states_to_goal()], label="States to goal", color="red", marker="x")
    # plt.scatter(minigrid_lmdp.states_to_goal(), minigrid_lmdp.V[minigrid_lmdp.states_to_goal()], label="States to goal", color="green", marker="x")
    # plt.legend()
    # plt.show()
    
    # states_to_goal_sotchastic = []
    # for i in range(10):
    #     states_to_goal_sotchastic.append(minigrid_lmdp.states_to_goal(stochastic=True))
    
    
    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(np.arange(len(states_to_goal_ideal)), states_to_goal_ideal, linewidth=2, label="Ideal")
    # for states in states_to_goal_sotchastic:
    #     plt.plot(np.arange(len(states)), states, marker="x", color="gray", linewidth=0.3, markersize=3)
    # plt.xlabel("Step number")
    # plt.ylabel("State idx")
    # plt.grid()
    # plt.legend()
    # plt.show()
    
    # minigrid_lmdp.visualize_policy([(0, minigrid_lmdp.policy)])
    
    
    # visualize_stochasticity_rewards_embedded_lmdp(0, map=Maps.SIMPLE_TEST)
    # minigrid_lmdp.visualize_policy()
    # print(minigrid_lmdp.R)
    
    # minigrid_lmdp.visualize_policy(num_times=1, save_gif=False, save_path=f"assets/cliff_lmdp_stochastic_{stochastic}.gif")
    # # print("LMDP", minigrid_lmdp.V)
    
    # benchmark_mdp2lmdp_embedding(
    #     map=Maps.CLIFF,
    #     name=f"CLIFF_Stochasticity_{stochastic}_RewardCliff_700",
    #     allowed_actions=[
    #         MinigridActions.ROTATE_LEFT,
    #         MinigridActions.ROTATE_RIGHT,
    #         MinigridActions.FORWARD,
    #         # MinigridActions.PICKUP,
    #         # MinigridActions.DROP,
    #         # MinigridActions.TOGGLE,
    #     ]
    # )

    # gridworld_mdp.compute_value_function()
    # print(gridworld_mdp.policy)
    
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

    # for size in [10, 20, 30, 40, 45, 50]:
    #     print(f"Grid size: {size}")
    #     benchmark_lmdp2mdp_embedding(savefig=True, grid_size=size)
    # benchmark_lmdp2mdp_embedding(savefig=True, map=Maps.CLIFF, name="CLIFF")
    
    

    
    
    # embedded_mdp = minigrid_lmdp.to_MDP()

    # print(minigrid_mdp.P[:, :, 31])
    # grid_size = 30
    # benchmark_lmdp2mdp_embedding(savefig=True, grid_size=grid_size)
    # benchmark_lmdp2mdp_embedding(savefig=True, grid_size=grid_size, map=Maps.DOUBLE_DOOR, objects=Maps.DOUBLE_DOOR_OBJECTS, name="DOUBLE_DOOR")
    
    
    
    # embedded_mdp = minigrid_lmdp.to_MDP()
    
    # minigrid_lmdp.visualize_policy(save_gif=True, save_path="assets/original_lmdp.gif")

    
    
    # minigrid_mdp = MinigridMDP(
    #     grid_size=grid_size,
    #     map=Maps.DOUBLE_DOOR,
    #     allowed_actions=[
    #         MinigridActions.ROTATE_LEFT,
    #         MinigridActions.ROTATE_RIGHT,
    #         MinigridActions.FORWARD,
    #         MinigridActions.PICKUP,
    #         MinigridActions.DROP,
    #         MinigridActions.TOGGLE,
    #     ],
    #     objects=Maps.DOUBLE_DOOR_OBJECTS,
    #     mdp=embedded_mdp,
    #     deterministic=False
    # )
    # minigrid_mdp.visualize_policy(save_gif=True, save_path="assets/embedded_mdp.gif")

    
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
    
    