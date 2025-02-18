from domains.grid_world import GridWorldMDP, GridWorldPlotter, GridWorldLMDP
from domains.minigrid_env import MinigridMDP, MinigridActions, MinigridLMDP, MinigridLMDP_TDR
from algorithms import QLearning, QLearningPlotter, QLearningHyperparameters, QLearningHyperparameterExplorer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
import numpy as np
from utils.maps import Maps
from utils.benchmarks import benchmark_value_iteration, benchmark_parallel_p, benchmark_lmdp2mdp_embedding
from minigrid.manual_control import ManualControl
from custom_palette import CustomPalette
from utils.stats import ValueIterationStats

if __name__ == "__main__":
    # benchmark_parallel_p()
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

    # for size in [10, 20, 30, 40, 45, 50]:
    #     print(f"Grid size: {size}")
    #     benchmark_lmdp2mdp_embedding(savefig=True, grid_size=size)
    # benchmark_lmdp2mdp_embedding(savefig=True, map=Maps.CLIFF, name="CLIFF")
    
    
    # minigrid_lmdp.visualize_policy(save_gif=False, save_path="assets/original_lmdp.gif")
    
    # embedded_mdp = minigrid_lmdp.to_MDP()
    
    minigrid_mdp = MinigridMDP(
        # grid_size=60,
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
        # threads=2
        # sparse_optimization=False
    )

    # print(minigrid_mdp.P[:, :, 31])
    
    minigrid_lmdp = MinigridLMDP(
        grid_size=2,
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
        threads=4,
        sparse_optimization=True
    )
    
    
    # print(minigrid_lmdp.num_states)
    # print(minigrid_mdp.num_states)
    
    # print(minigrid_lmdp.num_actions)
    # print(minigrid_mdp.num_actions)
    
    # print(minigrid_lmdp.num_non_terminal_states)
    # print(minigrid_mdp.num_non_terminal_states)
    
    # equal = []
    # for state in range(minigrid_lmdp.num_non_terminal_states):
    #     mdps = set()
    #     for action in range(minigrid_mdp.num_actions):
    #         mdps.add(int(np.where(minigrid_mdp.P[state, action, :] != 0)[0][0]))
        
    #     lmdps = set([int(a) for a in np.where(minigrid_lmdp.P[state, :] != 0)[0]])
    #     equal.append(lmdps == mdps)
    #     print(f"MDP: {mdps}, LMDP: {lmdps}, {lmdps == mdps}")
    
    # minigrid_mdp.value_iteration()
    # print(minigrid_mdp.R)
    # vs = []
    # for i in range(1, 30):
    #     V, _ = minigrid_mdp.value_iteration(max_iter=i)
    #     vs.append(V)
    
    
    minigrid_mdp.visualize_policy(save_gif=True, save_path=f"assets/mdppolicy.gif", num_times=1)
    minigrid_lmdp.visualize_policy(save_gif=True, save_path=f"assets/lmdppolicy.gif", num_times=1)
    # V, vs, _ = minigrid_mdp.value_iteration(max_iter=100)
    # policy = minigrid_mdp.get_optimal_policy(V)
    # # # state = minigrid_mdp.minigrid_env.custom_grid.state_index_mapper[2670]
    # # # while True:
    # # #     state, _, _ = minigrid_mdp.move(state, policy[minigrid_mdp.minigrid_env.custom_grid.states.index(state)])
    # # #     print(f"State: {minigrid_mdp.minigrid_env.custom_grid.states.index(state)}")
    # # # print(len(vs))



    # frame_files = []
    # for i, v in enumerate(vs):
    #     fig, ax = plt.subplots(figsize=(10, 5))

    #     ax.plot(np.arange(len(v)), v, color="blue")        
    #     plt.ylim(-100, 0)
    #     plt.title(f"Iteration {i}")
    #     plt.grid()
    #     plt.xlabel("States")
    #     plt.ylabel("Value Function")
    #     frame_file = f"assets/frame_{i}.png"
    #     plt.savefig(frame_file, dpi=300)
    #     frame_files.append(frame_file)
        
        
    #     plt.close(fig)

    # # Create a GIF using PIL
    # frames = []
    # for frame_file in frame_files:
    #     frame = Image.open(frame_file)
    #     frames.append(frame)

    # # Save the frames as a GIF
    # frames[0].save("assets/value_iteration_evolution_incorrect.gif", save_all=True, append_images=frames[1:], duration=250, loop=0)

    # # Optionally, delete the temporary frame images to save space
    # for frame_file in frame_files:
    #     os.remove(frame_file)
            
    
    
        
    # print(np.where(minigrid_mdp.P[4, :, :] != 0))
    # for i in range(minigrid_mdp.P.shape[0]):
    #     # print(i)
    #     print(np.where(minigrid_mdp.P[i, :] != 0))
    # print(minigrid_mdp.num_non_terminal_states)
    # print(np.where(minigrid_mdp.R[:minigrid_mdp.num_non_terminal_states, :] >= 0))
    # print(np.where(minigrid_mdp.R[minigrid_mdp.num_non_terminal_states:, :] < 0))
    # for i in range(minigrid_mdp.P.shape[0]):
    #     print(np.sum(minigrid_mdp.P[i, :, :], axis=1))
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
    
    