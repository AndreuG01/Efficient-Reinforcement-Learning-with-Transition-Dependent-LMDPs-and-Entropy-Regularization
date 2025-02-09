import numpy as np
from models.MDP import MDP
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product
import random
import os
import pickle
import concurrent.futures
from custom_palette import CustomPalette


class QLearning:
    """
    Implementation of the Q-Learning algorithm as described in Sutton and Barto's "Reinforcement Learning: An Introduction" (Page 131).

    Attributes:
        mdp (MDP): The Markov Decision Process (MDP) environment.
        Q (np.ndarray): The Q-value table of shape (num_states, num_actions).
        alpha (float): The learning rate, controls how much newly acquired information overrides old information.
        gamma (float): The discount factor, controls the importance of future rewards.
        epsilon (float): Exploration rate, defines the probability of taking a random action.
        epsilon_decay (float): The rate at which the exploration factor decays after each episode.
        curr_state (int): The current state of the agent in the MDP.
        reward (float): The reward obtained in the current step.
        episode_terminated (bool): Flag indicating whether the current episode has terminated.
        info_every (int): Frequency of logging training progress (in steps).
    """

    def __init__(
        self,
        mdp: MDP,
        alpha: float,
        gamma: float,
        epsilon: float = 1.0,
        info_every: int = 1000,
        epsilon_decay: float = 0.995,
        alpha_decay: int = 0
    ):
        """
        Initializes the Q-Learning agent with the given parameters.

        Args:
            mdp (MDP): The Markov Decision Process environment.
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            epsilon (float, optional): Exploration rate. Defaults to 1.0.
            info_every (int, optional): Log training progress every 'info_every' steps. Defaults to 1000.
            epsilon_decay (float, optional): Decay rate of epsilon. Defaults to 0.995.
        """
        self.mdp = mdp
        self.Q = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        self.Q[self.mdp.num_non_terminal_states, :] = self.mdp.R[self.mdp.num_non_terminal_states, :]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.curr_state = self.mdp.s0
        self.reward = 0
        self.episode_terminated = False
        self.info_every = info_every
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay
        self.curr_epoch = 0
        
        if not hasattr(self.mdp, "V"):
            print("Computing optimal value function")
            self.mdp.compute_value_function()
        
        self.V_optimal = self.mdp.V
            
        

    def __take_action(self, state: int) -> int:
        """
        Chooses an action based on the epsilon-greedy policy.

        Args:
            state (int): The current state.

        Returns:
            int: The chosen action.
        """
        # TODO: this is epsilon greedy. Implemenent something such as softmax, which is more fairly comparable to that of an LMDP. Read notion document.
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.mdp.num_actions)
        return np.argmax(self.Q[state, :])

    def __step(self):
        """
        Performs a single step in the environment, updates the Q-values, and transitions to the next state.
        """
        action = self.__take_action(self.curr_state)
        next_state, self.reward, is_terminal = self.mdp.transition(action, self.curr_state)

        self.Q[self.curr_state, action] += self.alpha * (
            self.reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[self.curr_state, action]
        )

        self.curr_state = next_state

        if is_terminal:
            self.curr_state = self.mdp.s0
            self.episode_terminated = True
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            if self.alpha_decay != 0:
                self.alpha = self.alpha_decay / (self.alpha_decay + self.curr_epoch)

    def train(self, num_steps: int, multiple_actions: bool = False, multiple_policies: bool = False) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], list[float], list[float]]:
        """
        Trains the agent by performing the Q-learning algorithm for a specified number of steps.

        Args:
            num_steps (int): The number of steps to train.
            multiple_actions (bool, optional): If True, the policy will allow multiple optimal actions for each state. Defaults to False.
            multiple_policies (bool, optional): If True, every self.info_every training epochs, a policy will be computed and stored

        Returns:
            tuple[np.ndarray, np.ndarray, list[float]]:
                - Q (np.ndarray): The learned Q-value table.
                - policies (list[tuple[int, np.ndarray]]): The derived policies during different episodes of the training process. It will contain only one policy if multiple_policies = False. Each element contains [epoch, policy]
                - rewards (list[float]): A list of cumulative rewards per episode.
                - errors (list[float]): A list of the error between the optimal value function and the estimated value function at each epoch.
        """
        self.curr_epoch = 0
        cumulative_reward = 0
        rewards = []
        episode_start = 0

        policies = []
        errors = []

        while self.curr_epoch < num_steps:
            self.__step()
            cumulative_reward += self.reward
            self.curr_epoch += 1
            
            V_estimate = self.get_value_function()
            errors.append(np.mean(np.square(self.V_optimal - V_estimate))) # Mean squared error between the optimal value function and the current estimated one.
            

            if self.episode_terminated:
                rewards[episode_start:self.curr_epoch] = [cumulative_reward] * (self.curr_epoch - episode_start + 1)
                episode_start = self.curr_epoch
                cumulative_reward = 0
                self.episode_terminated = False

            if self.curr_epoch % self.info_every == 0:
                print(f"Epoch [{self.curr_epoch} / {num_steps}]. Cumulative reward last episode: {rewards[episode_start - 1] if len(rewards) > 1 else 'not finished'}")
                if multiple_policies:
                    policies.append((self.curr_epoch, self.get_policy(multiple_actions=multiple_actions)))

        if not multiple_policies:
            policies.append((num_steps, self.get_policy(multiple_actions=multiple_actions)))

        if not self.episode_terminated:
            rewards[episode_start:self.curr_epoch] = [cumulative_reward] * (self.curr_epoch - episode_start)

        return self.Q, policies, rewards, errors

    def get_policy(self, multiple_actions: bool) -> np.ndarray:
        """
        Derives the optimal policy from the Q-value table.

        Args:
            multiple_actions (bool): If True, allows multiple optimal actions for each state.

        Returns:
            np.ndarray: The derived policy.
        """
        if multiple_actions:
            policy = np.empty(self.Q.shape[0], dtype=object)
            for s in range(self.Q.shape[0]):
                max_value = np.max(self.Q[s, :])
                policy[s] = [a for a, q in enumerate(self.Q[s, :]) if q == max_value]
        else:
            policy = np.argmax(self.Q, axis=1)

        return policy


    def get_value_function(self):
        return np.max(self.Q, axis=1)


class QLearningHyperparameters:
    """
    Represents the hyperparameters used for Q-Learning.

    Attributes:
        alpha (float): The learning rate that determines the extent to which newly acquired information overrides old information.
        alpha_decay (int): The rate at which the learning rate decays over time.
        gamma (float): The discount factor that determines the importance of future rewards.
        latex (bool): Whether to format the string representation in LaTeX style (default: True).
    """

    def __init__(self, alpha: float, alpha_decay: int, gamma: float, latex: bool = True):
        """
        Initializes a QLearningHyperparameters instance.

        Args:
            alpha (float): The learning rate.
            alpha_decay (int): The rate of learning rate decay.
            gamma (float): The discount factor for future rewards.
            latex (bool, optional): Whether to use LaTeX formatting for the string representation. Defaults to True.
        """
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.latex = latex

    def __str__(self):
        """
        Returns a formatted string representation of the hyperparameters in latex style, if self.latex is True.
        """
        return "{} {}, {} {}, {} {}".format(
            '\\alpha =' if self.latex else 'Alpha:', self.alpha, 
            '\\alpha_\\text{decay} =' if self.latex else 'Alpha decay:', self.alpha_decay,
            '\\gamma =' if self.latex else 'Gamma:', self.gamma
        )




class QLearningPlotter:
    """
    Plots the results of Q-Learning experiments, including rewards and errors, 
    with support for multiple hyperparameter configurations.

    Attributes:
        save_path (str): The directory where the plots will be saved.
        domain_name (str): The name of the domain or problem being analyzed (used as the plot title).
    """
    def __init__(self, save_path: str, domain_name: str = ""):
        """
        Initializes a QLearningPlotter instance.

        Args:
            save_path (str): The path to save the plot.
            domain_name (str, optional): A descriptive name for the problem domain. Defaults to an empty string.
        """
        self.save_path = save_path
        self.domain_name = domain_name
    
    def plot(self, rewards: list[list[float]], errors: list[list[float]], hyperparameters: list[QLearningHyperparameters]):
        """
        Generates and saves a plot visualizing rewards and errors for various hyperparameter configurations.

        Args:
            rewards (list[list[float]]): A list of reward sequences, where each inner list corresponds 
                                         to a specific hyperparameter configuration.
            errors (list[list[float]]): A list of error sequences, where each inner list corresponds 
                                        to a specific hyperparameter configuration.
            hyperparameters (list[QLearningHyperparameters]): A list of hyperparameter configurations 
                                                              corresponding to the reward and error sequences.

        Notes:
            - The plot includes two subplots: one for errors and one for cumulative rewards.
            - The legend is dynamically created based on the hyperparameter configurations.
            - Results are saved as a PNG image in the specified save path.
        """
        custom_palette = CustomPalette()
        
        n_rows = 2
        n_cols = 3
        fig_size = (15, 8)
        
        
        fig = plt.figure(layout="constrained", figsize=fig_size)
        gs = GridSpec(n_rows, n_cols, figure=fig)

        
        ax0 = fig.add_subplot(gs[0, :2])
        ax1 = fig.add_subplot(gs[1, :2])

        
        ax2 = fig.add_subplot(gs[:, 2]) # For the legend
        ax2.axis('off')

        fig.suptitle(f"{self.domain_name}", fontsize=14, fontweight="bold")
        
        
        legend_labels = []
        legend_colors = []
        
        for i, (reward, error) in enumerate(zip(rewards, errors)):
            color = custom_palette[i] if i < len(custom_palette) else f"#{random.randint(0, 0xFFFFFF):06X}"
            
            
            ax0.plot(np.arange(len(error)), error, color=color, label=f"${str(hyperparameters[i])}$")  # First row, first column
            ax0.set_xlabel(r"$\text{Epoch}$")
            ax0.set_ylabel(r"$\text{MSE (log scale)}$")
            ax0.set_yscale("symlog")
            ax0.grid(visible=True)
            # ax0.legend(fontsize=5, loc="lower left")
            
            ax1.plot(np.arange(len(reward)), reward, color=color, label=f"${str(hyperparameters[i])}$")  # Second row, first column
            ax1.set_xlabel(r"$\text{Epoch}$")
            ax1.set_ylabel(r"$\text{Cumulative reward (log scale)}$")
            ax1.set_yscale("symlog")
            ax1.grid(visible=True)

            
            hyperparameter_label = f"${str(hyperparameters[i])}$"
            legend_labels.append(hyperparameter_label)
            legend_colors.append(color)

        
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in legend_colors]
        ax2.legend(handles=handles, labels=legend_labels, loc="center")


        plt.savefig(f"{self.save_path}/hyperparameter_exploration.png", dpi=300)



class QLearningHyperparameterExplorer:
    """
    Tests different combinations of Q-learning hyperparameters to determine which is the best one.

    Attributes:
        mdp (MDP): the MDP instance that wants to be solved.
        alphas (list[float]): the list of learning rates that want to be tested.
        alphas_decays (list[int]): the learning rates decrease that want to be tested.
        gammas (list[float]): the list of discount factor for future rewards that wants to be tested.
        epochs (int): the number of epochs for which the MDP will be trained.
        out_path (str): the path where the generated files and the plot will be stored.
        domain_name (str): the name of the MDP domain that is being trained.
        __q_plotter (QLearningPlotter): an instance of a Q-learning plotter, that is responsible of plotting the different errors and rewards obtained during training with each combination of hyperparamters.
    """
    def __init__(self, mdp: MDP, alphas: list[float], alphas_decays: list[int], gammas: list[float], epochs: int, out_path: str, domain_name: str):
        """
        Initializes a QLearningHyperparameterExplorer instance.

        Args:
            mdp (MDP): the MDP instance that wants to be solved.
            alphas (list[float]): the list of learning rates that want to be tested.
            alphas_decays (list[int]): the learning rates decrease that want to be tested.
            gammas (list[float]): the list of discount factor for future rewards that wants to be tested.
            epochs (int): the number of epochs for which the MDP will be trained.
            out_path (str): the path where the generated files and the plot will be stored.
            domain_name (str): the name of the MDP domain that is being trained.
        """
        self.mdp = mdp
        self.alphas = alphas
        self.alphas_decays = alphas_decays
        self.gammas = gammas
        
        self.epochs = epochs
        
        if not os.path.exists(out_path):
            print("Creating output directory: ", out_path)
            os.makedirs(out_path)
        elif not os.path.isdir(out_path):
            raise FileExistsError(f"Out path: {out_path} exists and is not a directory")
            
        self.out_path = out_path
        
        self.__q_plotter = QLearningPlotter(out_path, domain_name=domain_name)
        
        
    
    def _test_combination(self, alpha, alpha_decay, gamma) -> tuple[list[float], list[float], QLearningHyperparameters, float, int, float, int]:
        """
        Tests a specific combination of hyperparameters for Q-learning
        
        Args:   
            alpha (float): The learning rate, which determines the impact of new information on the existing knowledge.
            alpha_decay (float): The rate at which the learning rate decays over time.
            gamma (float): The discount factor, representing the importance of future rewards.

        Returns:
            tuple: A tuple containing:
                - reward (list[float]): The cumulative reward values recorded during training.
                - error (list[float]): The mean squared error values recorded during training.
                - hyperparameters (QLearningHyperparameters): A QLearningHyperparameters instance with the different hyperparameters tested
                - max_reward_local (float): The maximum reward observed during training.
                - max_reward_epoch_local (int): The epoch at which the maximum reward was observed.
                - min_error_local (float): The minimum error observed during training.
                - min_error_epoch_local (int): The epoch at which the minimum error was observed.
        """
        print(f"{os.getpid()}. Testing: Alpha = {alpha}, Alpha decay = {alpha_decay}, Gamma = {gamma}")
        q_learner = QLearning(
            self.mdp,
            alpha=alpha,
            gamma=gamma,
            info_every=500000,
            alpha_decay=alpha_decay  
        )
        _, _, reward, error = q_learner.train(num_steps=self.epochs)
        
        max_reward_local = np.max(reward)
        max_reward_epoch_local = np.argmax(reward)
        min_error_local = np.min(error)
        min_error_epoch_local = np.argmin(error)
        
        return reward, error, QLearningHyperparameters(alpha, alpha_decay, gamma), max_reward_local, max_reward_epoch_local, min_error_local, min_error_epoch_local
        
    
    def test_hyperparameters(self):
        """
        Tests various combinations of hyperparameters for Q-Learning and evaluates their performance. It stores the
        different errors and rewards obtained during training in case they want to be later used for other purposes.

        This method:
        - Iterates through all combinations of hyperparameters (alpha, alpha_decay, gamma).
        - Uses multiprocessing to speed up the evaluation of each combination.
        - Tracks and identifies the hyperparameter combination that produces:
            - The maximum reward.
            - The minimum error.

        Returns:
            None
        """
        rewards = []
        errors = []
        hyperparameters = []
        combinations = list(product(self.alphas, self.alphas_decays, self.gammas))
        max_reward, max_reward_epoch, max_reward_hyper = -np.inf, 0, (self.alphas[0], self.alphas_decays[0], self.gammas[0])
        min_error, min_error_epoch, min_error_hyper = np.inf, 0, (self.alphas[0], self.alphas_decays[0], self.gammas[0])
        
        self.mdp.compute_value_function()
        
        # Concurrent hyperparameter testing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            concurrent_comb = {
                executor.submit(self._test_combination, alpha, alpha_decay, gamma) : (alpha, alpha_decay, gamma) for alpha, alpha_decay, gamma in combinations
            }
            
            for exec in concurrent.futures.as_completed(concurrent_comb):
                reward, error, hyperparam, max_reward_local, max_reward_epoch_local, min_error_local, min_error_epoch_local = exec.result()

                if max_reward_local > max_reward:
                    max_reward = max_reward_local
                    max_reward_epoch = max_reward_epoch_local
                    max_reward_hyper = hyperparam
                    
                if min_error_local < min_error:
                    min_error = min_error_local
                    min_error_epoch = min_error_epoch_local
                    min_error_hyper = hyperparam
                
                rewards.append(reward)
                errors.append(error)
                hyperparameters.append(hyperparam)
        
        print("Storing infromation...")
        self.__store_data(hyperparameters, "hyperparamters.pkl")
        self.__store_data(errors, "errors.pkl")
        self.__store_data(rewards, "rewards.pkl")
    
        with open(os.path.join(self.out_path, "results.txt"), "w") as f:
            f.write(f"Min error: {min_error} at epoch {min_error_epoch}. Hyperparameters: {min_error_hyper}\n")
            f.write(f"Max reward: {max_reward} at epoch {max_reward_epoch}. Hyperparameters: {max_reward_hyper}")
        
        
        self.__q_plotter.plot(rewards, errors, hyperparameters)
            
    
    def __store_data(self, data, name: str):
        with open(os.path.join(self.out_path, name), "wb") as f:
            pickle.dump(data, f)