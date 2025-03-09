import numpy as np
import time
from utils.stats import ModelBasedAlgsStats
from abc import ABC, abstractmethod
from domains.grid import CellType, CustomGrid
from collections.abc import Callable
from utils.state import State
from tqdm import tqdm
from scipy.sparse import csr_matrix
import models

class MDP(ABC):
    """
    A class representing a Markov Decision Process (MDP).

    The MDP is defined by a 4-tuple: (S, A, P, R) where:
    - S: A finite set of states (num_states)
    - A: A finite set of actions (num_actions)
    - P: A state transition probability function P(s' | s, a)
    - R: A reward function R(s, a)

    Attributes:
    - num_states (int): The total number of states in the MDP.
    - num_terminal_states (int): The number of terminal states.
    - num_non_terminal_states (int): The number of non-terminal states.
    - num_actions (int): The number of actions available in the MDP.
    - s0 (int): The initial state index.
    - gamma (float): The discount factor (must be between 0 and 1).
    - P (np.ndarray): The state transition probability matrix.
    - R (np.ndarray): The reward matrix for each state-action pair.
    #TODO: complete when code is finished
    """

    def __init__(
        self,
        num_states: int,
        num_terminal_states: int,
        allowed_actions: list[int],
        gamma: int = 1,
        s0: int = 0,
        deterministic: bool = False
    ) -> None:
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
        self.deterministic = deterministic
        
        # Initialize transition probabilities and rewards to zero
        self.P = np.zeros((self.num_non_terminal_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        
    
    
    
    def generate_P(self, grid: CustomGrid, stochastic_prob: float = 0.9):
        """
        Generates the transition probability matrix (P) for the MDP, based on the dynamics of the environment
        (deterministic or stochastic).

        Args:
        - pos (dict[int, list]): The different positions of the grid
        - move (Callable): A function that determines the next state based on the current state and action.
        The function signature should be `move(state: State, action: int) -> tuple[next_state: State, reward: float, done: bool]`.
        - grid (CustomGrid): The grid environment for which the transition matrix is being generated.
        """
        pos = grid.states
        terminal_pos = grid.terminal_states
        print(f"Allowed actions {self.__alowed_actions}")
        for state in tqdm(range(self.num_non_terminal_states), desc="Generating transition matrix P", total=self.num_non_terminal_states):
            for action in self.__alowed_actions:
                # print(state, action)
                if grid.is_cliff(grid.state_index_mapper[state]):
                    # Cliff states always have the full probability to transition to the initial state, regardless of whether the model is stochastic, or deterministic
                    next_state = self.s0
                    self.P[state, action, next_state] = 1
                    continue
                else:
                    next_state, _, terminal = grid.move(pos[state], action)
                    # Convert from coordinate-like system (i, j) (grid format) to index based (idx) (matrix format)
                    if terminal:
                        next_state = len(pos) + terminal_pos.index(next_state)
                    else:
                        next_state = pos.index(next_state)

                if self.deterministic:
                    self.P[state, action, next_state] = 1
                else:
                    # Stochastic policy. With 90% take the correct action, with 10% uniformly take every other action
                    self.P[state, action, next_state] = stochastic_prob
                    other_actions = [a for a in self.__alowed_actions if a != action]
                    for new_action in other_actions:
                        
                        next_state, _, terminal = grid.move(pos[state], new_action)
                        if terminal:
                            next_state = len(pos) + terminal_pos.index(next_state)
                        else:
                            next_state = pos.index(next_state)
                        self.P[state, action, next_state] += (1 - stochastic_prob) / len(other_actions)
                    
                    
        
        print(f"Generated matrix P with {self.P.size:,} elements")
    
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

    def value_iteration_inefficient(self, epsilon=1e-5) -> tuple[np.ndarray, ModelBasedAlgsStats]:
        """
        Perform value iteration to compute the optimal value function.
        From Sutton and Barto, page 83 from my references PDF #TODO: remove in a future.

        Args:
        - epsilon (float, optional): The threshold for stopping the iteration (default is 1e-5).

        Returns:
        - V (np.ndarray): The optimal value function for each state.
        - ModelBasedAlgsStats: An object containing statistics about the value iteration process (time, rewards, deltas, etc.).
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
        
        return V, ModelBasedAlgsStats(elapsed_time, rewards, iterations, deltas, self.num_states)
    
    
    def value_iteration(self, epsilon=1e-10) -> tuple[np.ndarray, ModelBasedAlgsStats]:
        """
        Perform value iteration to compute the optimal value function.
        Efficiently implemented with matrix operations
        From Sutton and Barto, page 83 from my references PDF #TODO: remove in a future.

        Args:
        - epsilon (float, optional): The threshold for stopping the iteration (default is 1e-5).

        Returns:
        - V (np.ndarray): The optimal value function for each state.
        - ModelBasedAlgsStats: An object containing statistics about the value iteration process (time, rewards, deltas, etc.).
        """
        V = np.zeros(self.num_states)

        iterations = 0
        start_time = time.time()
        deltas = []
        Vs = []
        print(f"Value iteration...")

        while True:
            delta = 0
            # expected_values = np.tensordot(self.P, V, axes=((2), (0))) # num_non_teminal X num_actions
            expected_values = self.P @ V
            Q = self.R + self.gamma * np.concatenate((expected_values, self.R[self.num_non_terminal_states:, :])) # num_states X num_actions

            V_new =  np.max(Q, axis=1)
            Vs.append(V_new)
            delta = np.linalg.norm(V - V_new, np.inf)
            
            if iterations % 1000 == 0:
                print(f"Iter: {iterations}. Delta: {delta}")
            
            
            if delta < epsilon:
                break

            V = V_new
            iterations += 1
            deltas.append(delta)

        elapsed_time = time.time() - start_time
        print(f"Converged in {iterations} iterations")
        return V, ModelBasedAlgsStats(elapsed_time, iterations, deltas, self.num_states, Vs, "VI")
    
    
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
        expected_utilities = self.R[:self.num_non_terminal_states] + \
                     self.gamma * np.einsum("saj,j->sa", self.P[:self.num_non_terminal_states], V)
        if multiple_actions:
            max_values = np.max(expected_utilities, axis=1, keepdims=True)
            policy = (expected_utilities == max_values).astype(int)  # Binary mask for optimal actions
        else:
            policy = np.argmax(expected_utilities, axis=1)

        return policy
    
        
    def to_LMDP(self):
        print(f"Computing the LMDP embedding of this MDP...")
        
        
        lmdp = models.LMDP.LMDP(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            sparse_optimization=True,
            lmbda=1,
            s0=self.s0
        )
        
        if self.deterministic and False:
            pass
            
        else:
            epsilon = 1e-10
            for state in range(self.num_non_terminal_states):
                # print(f"STATE: {state}")
                B = self.P[state, :, :]
                zero_cols = np.all(B == 0, axis=0)
                zero_cols_idx = np.where(zero_cols)[0]
                
                # Remove 0 columns
                B = B[:, ~zero_cols]
                
                # If an element of B is zero, its entire column must be 0, otherwise, replace the problematic element by epsilon and renormalize
                B[B == 0] = epsilon
                B /= np.sum(B, axis=1).reshape(-1, 1)
                
                log_B = np.where(B != 0, np.log(B), B)
                y = self.R[state] + np.sum(B * log_B, axis=1)
                B_dagger = np.linalg.pinv(B)
                c = B_dagger @ y
                
                
                R = np.log(np.sum(np.exp(c)))
                x = c - R * np.ones(shape=c.shape)
                lmdp.R[state] = R
                lmdp.P[state, ~zero_cols] = np.exp(x)
                
        
        lmdp.R[self.num_non_terminal_states:] = np.sum(self.R[self.num_non_terminal_states:], axis=1) / self.num_actions
        z, _ = lmdp.power_iteration()
        V_lmdp = lmdp.get_value_function(z)
        V_mdp, _ = self.value_iteration()
        
        print("EMBEDDING ERROR:", np.mean(np.square(V_lmdp - V_mdp)))    
        return lmdp
    
    
    def to_LMDP_TDR(self):
        print(f"Computing the LMDP-TDR embedding of this MDP...")
        
        
        lmdp_tdr = models.LMDP_TDR.LMDP_TDR(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            sparse_optimization=True,
            lmbda=1,
            s0=self.s0
        )
        
        if self.deterministic and False:
            pass
            
        else:
            epsilon = 1e-10
            for state in range(self.num_non_terminal_states):
                # print(f"STATE: {state}")
                B = self.P[state, :, :]
                zero_cols = np.all(B == 0, axis=0)
                
                # Remove 0 columns
                B = B[:, ~zero_cols]
                
                # If an element of B is zero, its entire column must be 0, otherwise, replace the problematic element by epsilon and renormalize
                B[B == 0] = epsilon
                B /= np.sum(B, axis=1).reshape(-1, 1)

                log_B = np.where(B != 0, np.log(B), B)
                y = self.R[state] + np.sum(B * log_B, axis=1)
                B_dagger = np.linalg.pinv(B)
                x = B_dagger @ y
                
                support_x = [col for col in zero_cols if col == False]
                len_support = len(support_x)
                
                lmdp_tdr.R[state, ~zero_cols] = x + lmdp_tdr.lmbda * np.log(len_support)
                lmdp_tdr.P[state, ~zero_cols] = np.exp(-np.log(len_support))
                
        
        lmdp_tdr.R = csr_matrix(lmdp_tdr.R)
        lmdp_tdr.P = csr_matrix(lmdp_tdr.P)
        
        z = lmdp_tdr.power_iteration()
        V_lmdp = lmdp_tdr.get_value_function(z)
        V_mdp, _ = self.value_iteration()
        
        print("EMBEDDING ERROR:", np.mean(np.square(V_lmdp - V_mdp)))    
        return lmdp_tdr

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
