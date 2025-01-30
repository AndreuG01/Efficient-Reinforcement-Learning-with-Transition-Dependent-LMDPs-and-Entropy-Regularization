import numpy as np
import time
from utils.stats import ValueIterationStats
from abc import ABC, abstractmethod
from domains.grid import CellType, CustomGrid
from collections.abc import Callable
from utils.state import State

class MDP(ABC):
    """
    A class representing a Markov Decision Process (MDP).

    The MDP is defined by a 5-tuple: (S, A, T, R, gamma) where:
    - S: A finite set of states (num_states)
    - A: A finite set of actions (num_actions)
    - P: A state transition probability function P(s' | s, a)
    - R: A reward function R(s, a)
    - gamma: A discount factor (gamma)

    Attributes:
    - num_states (int): The total number of states in the MDP.
    - num_terminal_states (int): The number of terminal states.
    - num_non_terminal_states (int): The number of non-terminal states.
    - num_actions (int): The number of actions available in the MDP.
    - s0 (int): The initial state index.
    - gamma (float): The discount factor (must be between 0 and 1).
    - P (np.ndarray): The state transition probability matrix.
    - R (np.ndarray): The reward matrix for each state-action pair.
    """

    def __init__(self, num_states: int, num_terminal_states: int, allowed_actions: list[int], gamma: int = 1, s0: int = 0) -> None:
        """
        Initialize the MDP with the given parameters.

        Args:
        - num_states (int): The total number of states in the MDP.
        - num_terminal_states (int): The number of terminal states in the MDP.
        - num_actions (int): The number of actions available.
        - gamma (float, optional): The discount factor (default is 1).
        - s0 (int, optional): The initial state (default is 0).

        Raises:
        - AssertionError: If the initial state is not valid or the number of terminal states is greater than or equal to the total number of states.
        """
        assert 0 <= s0 <= num_states - 1, "Initial state must be a valid state"
        assert num_terminal_states < num_states, "There must be fewer terminal states than the total number of states"
        assert 0 <= gamma <= 1, "Discount factor must be in the range [0, 1]"
        
        self.num_states = num_states
        self.num_terminal_states = num_terminal_states
        self.num_non_terminal_states = self.num_states - self.num_terminal_states
        self.__alowed_actions = allowed_actions
        self.num_actions = len(self.__alowed_actions)
        self.s0 = s0
        self.gamma = gamma
        
        # Initialize transition probabilities and rewards to zero
        self.P = np.zeros((self.num_non_terminal_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))
        
    
    
    
    def generate_P(self, pos: dict[int, list], move: Callable, grid: CustomGrid):
        """
        Generates the transition probability matrix (P) for the grid world, based on the dynamics of the environment
        (deterministic or stochastic).

        Args:
        - pos (dict[int, list]): The different positions of the grid
        - move (Callable): A function that determines the next state based on the current state and action.
        The function signature should be `move(state: State, action: int) -> tuple[next_state: State, reward: float, done: bool]`.
        - grid (CustomGrid): The grid environment for which the transition matrix is being generated.
        """
        
        for state in range(self.num_non_terminal_states):
            for action in self.__alowed_actions:
                if grid.state_index_mapper[state] in pos[CellType.CLIFF]:
                    next_state = self.s0
                else:
                    next_state, _, _ = move(pos[CellType.NORMAL][state], action)
                    # Convert from coordinate-like system (i, j) (grid format) to index based (idx) (matrix format)
                    if next_state in pos[CellType.GOAL]:
                        next_state = pos[CellType.GOAL].index(next_state) + len(pos[CellType.NORMAL])
                    else:
                        next_state = pos[CellType.NORMAL].index(next_state)

                if self.deterministic:
                    self.P[state, action, next_state] = 1
                else:
                    # TODO: not tested for minigrid
                    # Stochastic policy. With 70% take the correct action, with 30% take a random action
                    self.P[state, action, next_state] = 0.7
                    rand_action = np.random.choice([a for a in self.__alowed_actions if a != action])
                    next_state, _, _ = move(pos[CellType.NORMAL][state], rand_action)
                    if next_state in pos[CellType.GOAL]:
                        next_state = pos[CellType.GOAL].index(next_state) + len(pos[CellType.NORMAL])
                    else:
                        next_state = pos[CellType.NORMAL].index(next_state)
                    self.P[state, action, next_state] += 0.3
    
    
    
    @abstractmethod
    def _generate_R(self):
        raise NotImplementedError("Implement in the subclass.")
    

    def transition(self, action: int, state: int) -> tuple[int, float, bool]:
        """
        Simulate a state transition given an action and current state.

        Args:
        - action (int): The action taken.
        - state (int): The current state.

        Returns:
        - next_state (int): The state reached after the transition.
        - reward (float): The reward obtained for the transition.
        - terminal (bool): True if the next state is a terminal state, False otherwise.
        """
        # With probability P, choose one of the next states
        next_state = np.random.choice(self.num_states, p=self.P[state, action])
        
        return (
            next_state,
            self.R[state, action],
            next_state >= self.num_non_terminal_states
        )

    def value_iteration(self, epsilon=1e-5) -> tuple[np.ndarray, ValueIterationStats]:
        """
        Perform value iteration to compute the optimal value function.
        From Sutton and Barto, page 83 from my references PDF #TODO: remove in a future.

        Args:
        - epsilon (float, optional): The threshold for stopping the iteration (default is 1e-5).

        Returns:
        - V (np.ndarray): The optimal value function for each state.
        - ValueIterationStats: An object containing statistics about the value iteration process (time, rewards, deltas, etc.).
        """
        V = np.zeros(self.num_states)
        iterations = 0
        cumulative_reward = 0
        rewards = []
        start_time = time.time()
        deltas = []

        while True:
            delta = 0
            for s in range(self.num_non_terminal_states):
                v = V[s]
                # TODO: Matrix multiplications
                action_rewards = [
                    sum(self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V[s_next])
                        for s_next in range(self.num_states))
                    for a in range(self.num_actions)
                ]
                best_action_reward = max(action_rewards)
                V[s] = best_action_reward
                cumulative_reward += best_action_reward
                delta = max(delta, abs(v - V[s]))
            
            rewards.append(cumulative_reward)
            deltas.append(delta)
            iterations += 1
            if delta < epsilon:
                break

        elapsed_time = time.time() - start_time
        
        return V, ValueIterationStats(elapsed_time, rewards, iterations, deltas, self.num_states)
    
    
    def compute_value_function(self):
        """
        Computes the value function using value iteration and extracts the optimal policy.
        """
        self.V, self.stats = self.value_iteration()
        
        self.policy = self.get_optimal_policy(self.V)
        self.policy_multiple_actions = self.get_optimal_policy(self.V, multiple_actions=True)
    
    

    def get_optimal_policy(self, V: np.ndarray, multiple_actions: bool = False) -> np.ndarray:
        """
        Derive the optimal policy from the value function.

        Args:
        - V (np.ndarray): The computed value function.

        Returns:
        - policy (np.ndarray): The optimal policy, where each element corresponds to the optimal action for a state.
        """
        policy = np.zeros(self.num_states, dtype=object)
        for s in range(self.num_non_terminal_states):  # Skip terminal states
            # Find the action that maximizes the expected utility
            vals = [
                sum(self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V[s_next])
                    for s_next in range(self.num_states))
                for a in range(self.num_actions)
            ]
            if multiple_actions:
                policy[s] = [i for i in range(len(vals)) if vals[i] == np.max(vals)]
            else:
                policy[s] = np.argmax(vals)
            
        return policy

    def print_rewards(self):
        """
        Print the rewards for each state-action pair.

        This method outputs the reward function R(s, a), showing the reward obtained by taking action a from state s.
        """
        print("State | Action | Reward")
        print("-" * 25)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                print(f"{s:5d} | {a:6d} | {self.R[s, a]:.3f}")

    def print_action_values(self, V):
        """
        Print the action values (Q(s, a)) for each state-action pair.

        This method computes Q(s, a) using the value function V and prints the values for each state-action pair.

        Args:
        - V (np.ndarray): The computed value function.
        """
        print("State | Action | Value")
        print("-" * 25)
        for s in range(self.num_non_terminal_states):  # Exclude terminal states
            for a in range(self.num_actions):
                q_value = sum(
                    self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V[s_next])
                    for s_next in range(self.num_states)
                )
                print(f"{s:5d} | {a:6d} | {q_value:.3f}")
