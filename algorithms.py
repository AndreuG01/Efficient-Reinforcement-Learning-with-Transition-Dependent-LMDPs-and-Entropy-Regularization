import numpy as np
from models.MDP import MDP
import matplotlib.pyplot as plt


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
        alpha_decay: float = None
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

    def __take_action(self, state: int) -> int:
        """
        Chooses an action based on the epsilon-greedy policy.

        Args:
            state (int): The current state.

        Returns:
            int: The chosen action.
        """
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
            if self.alpha_decay is not None:
                self.alpha = self.alpha_decay / (self.alpha_decay + self.curr_epoch)

    def train(self, num_steps: int, multiple_actions: bool = False, multiple_policies: bool = False) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], list[float]]:
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
        """
        self.curr_epoch = 0
        cumulative_reward = 0
        rewards = []
        episode_start = 0

        policies = []

        while self.curr_epoch < num_steps:
            self.__step()
            cumulative_reward += self.reward
            self.curr_epoch += 1

            if self.episode_terminated:
                rewards[episode_start:self.curr_epoch] = [cumulative_reward] * (self.curr_epoch - episode_start + 1)
                episode_start = self.curr_epoch
                cumulative_reward = 0
                self.episode_terminated = False

            if self.curr_epoch % self.info_every == 0:
                print(f"Epoch [{self.curr_epoch} / {num_steps}]. Cumulative reward last episode: {rewards[episode_start - 1]}")
                if multiple_policies:
                    policies.append((self.curr_epoch, self.get_policy(multiple_actions=multiple_actions)))

        if not multiple_policies:
            policies.append((num_steps, self.get_policy(multiple_actions=multiple_actions)))

        if not self.episode_terminated:
            rewards[episode_start:self.curr_epoch] = [cumulative_reward] * (self.curr_epoch - episode_start)

        return self.Q, policies, rewards

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



class QLearningHyperparameterExplorer:
    def __init__(self, mdp: MDP, alphas: list[float], alphas_decays: list[float]):
        raise NotImplementedError("Still not implemented")
        
    
    def test_hyperparameters(self):
        raise NotImplementedError("Still not implemented")