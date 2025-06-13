import numpy as np
import time
from utils.stats import ModelBasedAlgsStats, EmbeddingStats
from abc import ABC, abstractmethod
from domains.grid import CellType, CustomGrid
from collections.abc import Callable
from utils.state import State
from tqdm import tqdm
from scipy.sparse import csr_matrix
from models.LMDP_TDR import LMDP_TDR
from models.LMDP import LMDP
from copy import deepcopy
from typing import Literal
from joblib import Parallel, delayed, cpu_count
from utils.coloring import TerminalColor
import itertools
from .utils import compare_models
from utils.utils import print_overflow_message


class MDP(ABC):
    """
    A class representing a Markov Decision Process (MDP).

    The MDP is defined by a 5-tuple: (S, A, P, R, gamma) where:
        - S: A finite set of states (num_states)
        - A: A finite set of actions (num_actions)
        - P: A state transition probability function P(s' | s, a)
        - R: A reward function R(s, a)
        - gamma: A discount factor (0 <= gamma < 1).

    Attributes:
        num_states (int): The total number of states in the MDP.
        num_terminal_states (int): The number of terminal states.
        num_non_terminal_states (int): The number of non-terminal states.
        __allowed_actions (list[int]): A list of allowed actions in the MDP.
        num_actions (int): The number of actions available in the MDP.
        s0 (int): The initial state index.
        gamma (float): The discount factor (must be between 0 and 1).
        deterministic (bool): Whether the MDP is deterministic or stochastic.
        behavior (str): The behavior of the MDP, can be "deterministic", "stochastic", or "mixed".
        temperature (float): The temperature parameter in case it is an entropy-regularized MDP.
        policy_ref (np.ndarray): A reference policy for the MDP, used in entropy-regularized MDPs.
        P (np.ndarray): The state transition probability matrix.
        R (np.ndarray): The reward matrix for each state-action pair.
        verbose (bool): Whether to print verbose output during computations.
        dtype (np.dtype): The data type for the matrices, can be np.float32, np.float64, or np.float128.
    """

    def __init__(
        self,
        num_states: int,
        num_terminal_states: int,
        allowed_actions: list[int],
        gamma: int = 1,
        s0: int = 0,
        deterministic: bool = False,
        temperature: float = 0.0,
        behavior: Literal["deterministic", "stochastic", "mixed"] = "deterministic",
        verbose: bool = True,
        policy_ref: np.ndarray = None,
        dtype: np.dtype = np.float128
    ) -> None:
        """
        Initialize the MDP with the given parameters.

        Args:
            num_states (int): The total number of states in the MDP.
            num_terminal_states (int): The number of terminal states in the MDP.
            allowed_actions (list[int]): A list of allowed actions in the MDP.
            gamma (float, optional): The discount factor. Defaults to 1.
            s0 (int, optional): The initial state. Defaults to 0.
            deterministic (bool, optional): Whether the MDP is deterministic. Defaults to False.
            temperature (float, optional): The temperature parameter for entropy-regularized MDPs. Defaults to 0.0.
            behavior (str, optional): The behavior of the MDP, can be "deterministic", "stochastic", or "mixed". Defaults to "deterministic".
            verbose (bool, optional): Whether to print verbose output during computations. Defaults to True.
            policy_ref (np.ndarray, optional): A reference policy for the MDP, used in entropy-regularized MDPs. Defaults to None.
            dtype (np.dtype): The data type for the matrices, can be np.float32, np.float64, or np.float128.
            

        Raises:
        - AssertionError: If the initial state is not valid or the number of terminal states is greater than or equal to the total number of states.
        """
        assert 0 <= s0 <= num_states - 1, "Initial state must be a valid state"
        assert num_terminal_states < num_states, "There must be fewer terminal states than the total number of states"
        assert 0 <= gamma <= 1, "Discount factor must be in the range [0, 1]"
        assert behavior in ["deterministic", "stochastic", "mixed"], f"{behavior} behavior not supported."
        assert dtype in [np.float32, np.float64, np.float128], f"Only allowed data types: {[np.float32, np.float64, np.float128]}"
        
        self.dtype = dtype
        
        self.num_states = num_states
        self.num_terminal_states = num_terminal_states
        self.num_non_terminal_states = self.num_states - self.num_terminal_states
        self.__allowed_actions = allowed_actions
        self.num_actions = len(self.__allowed_actions)
        self.s0 = s0
        self.gamma = gamma
        self.deterministic = deterministic
        self.behavior = behavior
        self.temperature = temperature
        self.policy_ref = np.full((self.num_states, self.num_actions), 1.0 / self.num_actions, dtype=self.dtype) if policy_ref is None else policy_ref
        
        # Initialize transition probabilities and rewards to zero
        self.P = np.zeros((self.num_non_terminal_states, self.num_actions, self.num_states), dtype=self.dtype)
        self.R = np.zeros((self.num_states, self.num_actions), dtype=self.dtype)
        
        self.verbose = verbose
    

    def generate_P(self, grid: CustomGrid, stochastic_prob: float = 0.9, num_threads: int = 4, benchmark: bool = False) -> tuple[np.ndarray, float]:
        """
        Generates the transition probability matrix (P) for the LMDP, based on the dynamics of the environment.

        Args:
            grid (CustomGrid): The grid environment containing the states and their properties, as well as the movement logic.
            stochastic_prob (float): Probability of taking the intended action. Defaults to 0.9.
            actions (list[int]): List of actions that can be taken in the environment.
            num_threads (int): Number of threads to use for parallel processing. Defaults to 10.
            benchmark (bool): If True, measures the time taken to generate the transition matrix. Defaults to False.
        
        Returns:
            tuple[np.ndarray, float]: A tuple containing the transition probability matrix (P) and the time taken to generate it.
        """
        pos = grid.states
        terminal_pos = grid.terminal_states
        self._print(f"Allowed actions {self.__allowed_actions}")
        P = np.zeros((self.num_non_terminal_states, self.num_actions, self.num_states), dtype=self.dtype)
        
        def process_state(state: int) -> list[float]:
            row_updates = []
            for action in self.__allowed_actions:
                if grid.is_cliff(grid.state_index_mapper[state]):
                    # Cliff states always have the full probability to transition to the initial state, regardless of whether the model is stochastic, or deterministic
                    next_state = self.s0
                    row_updates.append((state, next_state, action, 1))
                    continue
                else:
                    next_state, _, terminal = grid.move(pos[state], action)
                    # Convert from coordinate-like system (i, j) (grid format) to index based (idx) (matrix format)
                    if terminal:
                        next_state = len(pos) + terminal_pos.index(next_state)
                    else:
                        next_state = pos.index(next_state)
                
                if self.deterministic:
                    row_updates.append((state, next_state, action, 1))
                else:
                    row_updates.append((state, next_state, action, stochastic_prob))
                    other_actions = [a for a in self.__allowed_actions if a != action]
                    for new_action in other_actions:
                        next_state, _, terminal = grid.move(pos[state], new_action)
                        if terminal:
                            next_state = len(pos) + terminal_pos.index(next_state)
                        else:
                            next_state = pos.index(next_state)
                        row_updates.append((state, next_state, action, (1 - stochastic_prob) / len(other_actions)))
            
            return row_updates
                    
        total_time = 0
        if benchmark: start_time = time.time()
        
        results = Parallel(n_jobs=min(num_threads, cpu_count()), temp_folder="/tmp")(
            delayed(process_state)(state) for state in tqdm(range(self.num_non_terminal_states),
                                                            desc="Generating transition matrix P",
                                                            total=self.num_non_terminal_states,
                                                            disable=not self.verbose)
        )
        
        for row_updates in results:
            for state_idx, next_state_idx, action, prob in row_updates:
                P[state_idx, action, next_state_idx] += prob
        
        if benchmark:
            end_time = time.time()
            total_time = end_time - start_time
        
        self._print(f"Generated matrix P with {self.P.size:,} elements")
        
        return P, total_time
    
    def _generate_R(self):
        raise NotImplementedError("Implement in the subclass.")
    

    def transition(self, action: int, state: int) -> tuple[int, float, bool]:
        """
        Simulate a transition in the MDP given the current state and action.
        
        Args:
            state (int): The current state index from which to transition.
            action (int): The action to take.
        
        Returns:
            tuple[int, float, bool]: A tuple containing:
                - next_state (int): The index of the next state after the transition.
                - reward (float): The reward received for transitioning to the next state.
                - is_terminal (bool): A boolean indicating whether the next state is a terminal state.
        """
        # With probability P, choose one of the next states
        next_state = np.random.choice(self.num_states, p=self.P[state, action] if self.dtype != np.float128 else self.P[state, action].astype(np.float64))
        
        return (
            next_state,
            self.R[state, action],
            next_state >= self.num_non_terminal_states
        )

    def value_iteration_inefficient(self, epsilon=1e-5) -> tuple[np.ndarray, ModelBasedAlgsStats]:
        """
        Perform value iteration to compute the optimal value function.
        This implementation is inefficient and uses a nested loop to compute the value function.

        Args:
            epsilon (float, optional): The threshold for stopping the iteration. Defaults to 1e-5.

        Returns:
            tuple[np.ndarray, ModelBasedAlgsStats]: A tuple containing:
                - V (np.ndarray): The optimal value function for each state.
                - ModelBasedAlgsStats: An object containing statistics about the value iteration process (time, rewards, deltas, etc.).
        """
        V = np.zeros(self.num_states, dtype=self.dtype)
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
        
        return V, ModelBasedAlgsStats(elapsed_time, rewards, iterations, deltas, self.num_states, descriptor="VI")
    
    
    def value_iteration(self, epsilon=1e-10, max_iterations=30000, temp: float = None) -> tuple[np.ndarray, ModelBasedAlgsStats, bool]:
        """
        Perform value iteration to compute the optimal value function.
        This implementation uses a more efficient approach with matrix operations to compute the value function.

        Args:
            epsilon (float, optional): The threshold for stopping the iteration. Defaults to 1e-5.
            max_iterations (int, optional): The maximum number of iterations to perform. Defaults to 30000.
            temp (float, optional): The temperature parameter for entropy-regularized MDPs. Defaults to None.
        
        Returns:
            tuple[np.ndarray, ModelBasedAlgsStats, bool]: A tuple containing:
                - V (np.ndarray): The optimal value function for each state.
                - ModelBasedAlgsStats: An object containing statistics about the value iteration process (time, iterations, deltas, etc.).
                - overflow (bool): A boolean indicating whether an overflow occurred during the computation.
        """
        
        if temp is not None:
            temperature = temp
        else:
            temperature = self.temperature
        
        V = np.zeros(self.num_states, dtype=self.dtype)

        iterations = 0
        start_time = time.time()
        deltas = []
        Vs = []
        self._print(f"Value iteration...")
        
        Q_old = np.zeros((self.num_states, self.num_actions), dtype=self.dtype)
        overflow = False

        while True:
            delta = 0
            # expected_values = np.tensordot(self.P, V, axes=((2), (0))) # num_non_teminal X num_actions
            expected_values = self.P @ V
            Q = self.R + self.gamma * np.concatenate((expected_values, self.R[self.num_non_terminal_states:, :])).astype(self.dtype) # num_states X num_actions

            if temperature == 0:
                V_new =  np.max(Q, axis=1).astype(self.dtype)
            else:
                exp_Q = self.policy_ref * np.exp(Q / temperature)
                if 0 in exp_Q:
                    print_overflow_message(exp_Q, self.policy_ref * np.exp(Q_old / temperature), self.dtype, temperature, target_value=0)
                    overflow = True
                    break
                V_new = (temperature * np.log(np.sum(exp_Q, axis=1))).astype(self.dtype)
            
            Vs.append(V_new)
            delta = np.linalg.norm(V - V_new, np.inf)
            
            if iterations % 500 == 0:
                self._print(f"Iter: {iterations}. Delta: {delta}")
            
            
            if delta < epsilon or iterations == max_iterations:
                break

            V = V_new
            self.Q = Q
            Q_old = Q
            iterations += 1
            deltas.append(delta)

        elapsed_time = time.time() - start_time
        if not overflow and iterations != max_iterations:
            self._print(f"Converged in {iterations} iterations")
        elif overflow:
            self._print(f"Not converged due to overflow.")
        else:
            self._print(f"Not converged due to max iterations reach.")
            
            
        return V, ModelBasedAlgsStats(elapsed_time, iterations, deltas, self.num_states, Vs, "VI"), overflow
    
    
    def compute_value_function(self, temp: float = None) -> None:
        """
        Computes the value function using value iteration and extracts the optimal policy.
        
        Args:
            temp (float, optional): The temperature parameter for entropy-regularized MDPs. Defaults to None.
            
        Returns:
            None
        """
        self.V, self.stats, _ = self.value_iteration(temp=temp)
        
        self.policy = self.get_optimal_policy(self.V, temp=temp)
        self.policy_multiple_actions = self.get_optimal_policy(self.V, multiple_actions=True, temp=temp)
    
    

    def get_optimal_policy(self, V: np.ndarray, multiple_actions: bool = False, temp: float = None) -> np.ndarray:
        """
        Derive the optimal policy from the value function.

        Args:
            V (np.ndarray): The computed value function.
            multiple_actions (bool): If True, allows multiple actions per state in the policy. Defaults to False.
            temp (float, optional): The temperature parameter for entropy-regularized MDPs. Defaults to None.

        Returns:
            policy (np.ndarray): The optimal policy, where each element corresponds to the optimal action for a state.
        """
        
        if temp is not None:
            temperature = temp
        else:
            temperature = self.temperature
        
        
        expected_utilities = self.Q
        
        if self.temperature == 0:
            max_values = np.max(expected_utilities, axis=1, keepdims=True)
            policy = (expected_utilities == max_values).astype(int)  # Binary mask for optimal actions
            if not multiple_actions:
                policy = policy.astype(self.dtype) / np.sum(policy, axis=1).reshape(-1, 1)
        else:
            policy = np.exp(expected_utilities / temperature) / np.sum(np.exp(expected_utilities / temperature), axis=1).reshape(-1,1)
        

        return policy
    
        
    def to_LMDP(self, lmbda: float = None) -> tuple[LMDP, bool]:
        """
        Convert the MDP into an LMDP using Todorov's method.
        
        Args:
            lmbda (float, optional): The regularization parameter for the LMDP. If None, defaults to 1. Defaults to None.
        
        Returns:
            tuple[LMDP, bool]: A tuple containing:
                - lmdp (LMDP): The LMDP representation of the MDP.
                - overflow (bool): A boolean indicating whether an overflow occurred during the computation.
        """
        self._print(f"Computing the LMDP embedding of this MDP...")
        
        
        lmdp = LMDP(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            sparse_optimization=True,
            # lmbda=self.temperature if self.temperature != 0 else 0.1, # TODO: change lmbda value when temperature is 0
            lmbda=1 if lmbda is None else lmbda,
            s0=self.s0,
            verbose=self.verbose,
            dtype=self.dtype
        )
        
        
        epsilon = 1e-10
        for state in range(self.num_non_terminal_states):
            B = self.P[state, :, :]
            zero_cols = np.all(B == 0, axis=0)
            zero_cols_idx = np.where(zero_cols)[0]
            
            # Remove 0 columns
            B = B[:, ~zero_cols]
            
            # If an element of B is zero, its entire column must be 0, otherwise, replace the problematic element by epsilon and renormalize
            B[B == 0] = epsilon
            B /= np.sum(B, axis=1).reshape(-1, 1).astype(self.dtype)
            
            log_B = np.where(B != 0, np.log(B), B)
            v = self.R[state] + lmdp.lmbda * np.sum(B * log_B, axis=1)
            B_dagger = np.linalg.pinv(B.astype(np.float64))
            x = B_dagger @ v
            
            if lmdp.lmbda != 0 and self.deterministic:
                for i, next_state in enumerate(np.where(zero_cols == False)[0]):
                    res = 0
                    for next_action in np.where(self.P[state, :, next_state] != 0)[0]:
                        res += self.policy_ref[state, next_action] * np.exp(self.R[state, next_action] / lmdp.lmbda)
                    res = lmdp.lmbda * np.log(res)
                    x[i] = res
            
            R = lmdp.lmbda * np.log(np.sum(np.exp(x / lmdp.lmbda)))
            lmdp.R[state] = R
            lmdp.P[state, ~zero_cols] = np.exp((x - R * np.ones(shape=x.shape, dtype=self.dtype)) / lmdp.lmbda)
                
        lmdp.R[self.num_non_terminal_states:] = np.sum(self.R[self.num_non_terminal_states:], axis=1) / self.num_actions
        z, lmdp.stats, overflow = lmdp.power_iteration()
        
        if not overflow:
            lmdp.V = lmdp.get_value_function(z)
            V_mdp, stats, _ = self.value_iteration(temp=lmdp.lmbda)
            
            if not hasattr(self, "stats"):
                self.stats = stats
            if not hasattr(self, "V"):
                self.V = V_mdp
            
            self._print(f"EMBEDDING ERROR MDP to LMDP: {np.mean(np.square(lmdp.V - V_mdp))}")
        
        return lmdp, overflow
    
    
    def _ternary_search_lambda(self, low: float, high: float, stats: EmbeddingStats, tol: float = 1e-4, max_iter: int = 100) -> float:
        """
        Perform ternary search to find the best lambda value within a given range.

        Args:
            low (float): The lower bound of the search range.
            high (float): The upper bound of the search range.
            stats (EmbeddingStats): The statistics object to store information about the search.
            tol (float): The tolerance for stopping the search.
            max_iter (int): The maximum number of iterations.

        Returns:
            float: The best lambda value found.
        """
        initial_low = low
        initial_high = high
        bar_length = 100

        for _ in range(max_iter):
            if high - low < tol:
                # To ensure that the progress bar ends cleanly
                self._print("")
                break

            mid_1 = low + (high - low) / 3
            mid_2 = high - (high - low) / 3
            
            self._print_lambda_progress(low, high, initial_low, initial_high, bar_length)
    
            err_1, _ = self._compute_lambda_error(mid_1)
            err_2, _ = self._compute_lambda_error(mid_2)
            
            err_left, _ = self._compute_lambda_error(low)
            err_right, _ = self._compute_lambda_error(high)
            stats.add_ternary_search_info(mid_1, err_1, mid_2, err_2, low, err_left, high, err_right)

            if err_1 < err_2:
                high = mid_2
                error, _ = self._compute_lambda_error(mid_1)
                stats.add_optimal_ternary_search_info(mid_1, error)
            else:
                low = mid_1
                error, _ = self._compute_lambda_error(mid_2)
                stats.add_optimal_ternary_search_info(mid_2, error)
                

        return (low + high) / 2


    def _compute_lambda_error(self, lmbda: float) -> tuple[float, bool]:
        """
        Compute the embedding error for a given lambda value.

        Args:
            lmbda (float): The lambda value to evaluate.

        Returns:
            float: The embedding error.
            bool: Whether lmbda generates overflow or not
        """
        verbose = self.verbose
        self.verbose = False
        lmdp, _, overflow = self.to_LMDP_TDR(lmbda=lmbda, find_best_lmbda=False)
        if not overflow:
            lmdp.compute_value_function()
        
        self.verbose = verbose
        return np.mean(np.square(self.V - lmdp.V)) if not overflow else -1, overflow


    def _print_lambda_progress(self, low: float, high: float, initial_low: float, initial_high: float, bar_length: int = 100) -> None:
        """
        Print the progress of the lambda refinement process.

        Args:
            low (float): The current lower bound of the search range.
            high (float): The current upper bound of the search range.
            initial_low (float): The initial lower bound of the search range.
            initial_high (float): The initial upper bound of the search range.
            bar_length (int): The length of the progress bar.
        """
        low_pos = int(bar_length * (low - initial_low) / (initial_high - initial_low))
        high_pos = int(bar_length * (high - initial_low) / (initial_high - initial_low))

        low_pos = max(0, min(low_pos, bar_length - 1))
        high_pos = max(0, min(high_pos, bar_length - 1))

        bar = ["-" if i < low_pos or i > high_pos else " " for i in range(bar_length)]
        bar[0] = TerminalColor.colorize("|", "purple", bold=True)
        bar[-1] = TerminalColor.colorize("|", "purple", bold=True)
        if low_pos != high_pos:
            bar[low_pos] = TerminalColor.colorize("[", "green")
            bar[high_pos] = TerminalColor.colorize("]", "green")
        else:
            if low_pos == 0:
                bar[1] = TerminalColor.colorize("]", "green")
                bar[high_pos] = TerminalColor.colorize("[", "green")
            else:
                bar[low_pos-1] = TerminalColor.colorize("[", "green")
                bar[high_pos] = TerminalColor.colorize("]", "green")

        bar_display = "".join(bar)
        msg = f"Refining temperature: {bar_display}  Current range: {round(low, 3)} - {round(high, 3)}"
        self._print(msg.ljust(120), end="\r")


    def _find_best_lambda(self, stats: EmbeddingStats) -> float:
        """
        Find the best lambda value for the LMDP embedding using a combination of iterative search and ternary search.
        
        Args:
            stats (EmbeddingStats): The statistics object to store information about the search process.
            
        Returns:
            float: The best lambda value found.
        """
        start_lmbda = max(0.05, self.temperature)
        lmbda = start_lmbda
        step = 1

        tried = []
        errors = []
        prev_error = None

        num_tries = 0
        spinner = itertools.cycle([
            "[=    ]", "[==   ]", "[===  ]", "[ ====]", "[  ===]", "[   ==]", "[    =]", "[     ]"
        ])
        spin = next(spinner)

        while True:
            if num_tries % 2 == 0:
                spin = next(spinner)
            if num_tries % 5 == 0:
                current_message = f"Testing lambda: {TerminalColor.colorize(str(lmbda), 'blue').ljust(100)}"

            self._print(f"{spin} {current_message}", end="\r")

            error, overflow = self._compute_lambda_error(lmbda)
            if overflow:
                # TODO: The lambda chosen is too big. Need to make it smaller and reduce the step size.
                print("overflow")
                step = max(0, step - 0.5)
                lmbda = tried[-1] + step
                continue
            stats.add_linear_search_info(lmbda, error)

            tried.append(lmbda)
            errors.append(error)

            if prev_error is not None:
                if error > prev_error:
                    # To ensure that the next message does not overwrite the current ones
                    self._print("")
                    break

            prev_error = error
            lmbda = round(lmbda + step, 3)
            num_tries += 1

        best_idx = np.argmin(errors)
        low = tried[best_idx]
        high = tried[best_idx + 1]

        final_lmbda = self._ternary_search_lambda(low, high, stats)
        error_optimal, _ = self._compute_lambda_error(final_lmbda)
        stats.set_optimal_parameter(final_lmbda, error_optimal)

        return final_lmbda

    
    def to_LMDP_TDR_iterative(self, lmbda: float = None, find_best_lmbda: bool = True) -> tuple[LMDP_TDR, EmbeddingStats, bool]:
        """
        Convert the MDP into an LMDP with transition-dependent rewards using the adaptation of Todorov's method derived in this thesis.
        
        This is an inefficient implementation that uses a nested loop to compute the LMDP-TDR representation.
        
        Args:
            lmbda (float, optional): The regularization parameter for the LMDP. If None, defaults to 1. Defaults to None.
            find_best_lmbda (bool): If True, finds the best lambda value for the LMDP embedding. Defaults to True.
        
        Returns:
            tuple[LMDP_TDR, EmbeddingStats, bool]: A tuple containing:
                - lmdp_tdr (LMDP_TDR): The LMDP-TDR representation of the MDP.
                - stats (EmbeddingStats): Statistics about the embedding process.
                - overflow (bool): A boolean indicating whether an overflow occurred during the computation.
        """
        self._print(f"Computing the LMDP-TDR embedding of this MDP...")
        stats = EmbeddingStats("iterative")
        stats.start_time()
        
        if find_best_lmbda:
            if np.all(np.sum(self.P == 1, axis=2)):
                # If the transition probability matrix is deterministic, skip the process of finding the optimal temperature lambda,
                # as we already know that lambda^* = beta
                print(f"{TerminalColor.colorize('WARNING: ', 'red', 'bold')} Will not look for optimal lambdas because MDP is deterministic. Best lambda is beta.")
            else:
                # Otherwise, look for the best temperature
                lmbda = self._find_best_lambda(stats=stats)
                self._print(TerminalColor.colorize(f"Best lmbda for LMDP has been: {round(lmbda, 3)}", "green", bold=True))
        
        lmdp_tdr = LMDP_TDR(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            sparse_optimization=True,
            lmbda=self.temperature if lmbda is None else lmbda,
            s0=self.s0,
            verbose=self.verbose,
            dtype=self.dtype
        )
        epsilon = 1e-10
        for state in range(self.num_non_terminal_states):
            B = self.P[state, :, :]
            zero_cols = np.all(B == 0, axis=0)
            
            # Remove 0 columns
            B = B[:, ~zero_cols]
            
            # If an element of B is zero, its entire column must be 0, otherwise, replace the problematic element by epsilon and renormalize
            B[B == 0] = epsilon
            B /= np.sum(B, axis=1).reshape(-1, 1)

            log_B = np.where(B != 0, np.log(B), B)
            y = self.R[state] - self.temperature * np.log(1 / self.policy_ref[state, :]) + lmdp_tdr.lmbda * np.sum(B * log_B, axis=1)
            B_dagger = np.linalg.pinv(B.astype(np.float64))
            x = B_dagger @ y
            
            support_x = [col for col in zero_cols if col == False]
            len_support = len(support_x)
            
            lmdp_tdr.R[state, ~zero_cols] = x + lmdp_tdr.lmbda * np.log(len_support)
            
            # assert all(lmdp_tdr.R[state, ~zero_cols] <= 0), f"Not all rewards for origin state {state} are negative:\n{lmdp_tdr.R[state, ~zero_cols]}"
            lmdp_tdr.P[state, ~zero_cols] = np.exp(-np.log(len_support))
                
        lmdp_tdr.R = csr_matrix(lmdp_tdr.R)
        lmdp_tdr.P = csr_matrix(lmdp_tdr.P)
        
        stats.end_time()
        
        z, _, overflow = lmdp_tdr.power_iteration()
        
        if not overflow:
            V_lmdp = lmdp_tdr.get_value_function(z)
            if not hasattr(self, "V"):
                self.V, self.stats, _ = self.value_iteration()
            # V_mdp, _, _ = self.value_iteration(temp=lmdp_tdr.lmbda)
            
            self._print(f"EMBEDDING ERROR: {np.mean(np.square(V_lmdp - self.V))}")
        
        return lmdp_tdr, stats, overflow
    
    def to_LMDP_TDR_vectorized(self, lmbda: float = None, find_best_lmbda: bool = True) -> tuple[LMDP_TDR, EmbeddingStats, bool]:
        """
        Convert the MDP into an LMDP with transition-dependent rewards using the adaptation of Todorov's method derived in this thesis.
        
        This approach is more efficient than the iterative one, as it uses vectorized operations to compute the embedding.
        
        Args:
            lmbda (float, optional): The regularization parameter for the LMDP. If None, defaults to 1. Defaults to None.
            find_best_lmbda (bool): If True, finds the best lambda value for the LMDP embedding. Defaults to True.
        
        Returns:
            tuple[LMDP_TDR, EmbeddingStats, bool]: A tuple containing:
                - lmdp_tdr (LMDP_TDR): The LMDP-TDR representation of the MDP.
                - stats (EmbeddingStats): Statistics about the embedding process.
                - overflow (bool): A boolean indicating whether an overflow occurred during the computation.
        """
        self._print(f"Computing the LMDP-TDR embedding of this MDP...")
        stats = EmbeddingStats("vectorized")
        stats.start_time()
        
        if find_best_lmbda:
            if np.all(np.sum(self.P == 1, axis=2)):
                # If the transition probability matrix is deterministic, skip the process of finding the optimal temperature lambda,
                # as we already know that lambda^* = beta
                print(f"{TerminalColor.colorize('WARNING: ', 'red', 'bold')} Will not look for optimal lambdas because MDP is deterministic. Best lambda is beta.")
            else:
                # Otherwise, look for the best temperature
                lmbda = self._find_best_lambda(stats=stats)
                self._print(TerminalColor.colorize(f"Best lmbda for LMDP has been: {round(lmbda, 3)}", "green", bold=True))
        
        lmdp_tdr = LMDP_TDR(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            sparse_optimization=True,
            lmbda=self.temperature if lmbda is None else lmbda,
            s0=self.s0,
            verbose=self.verbose,
            dtype=self.dtype
        )
        
        epsilon = 1e-10
        possible_transitions = np.any(self.P != 0, axis=1)
        num_possible_transitions = np.unique(np.sum(possible_transitions, axis=1))
        
        for num in num_possible_transitions:
            states = np.where(np.sum(possible_transitions, axis=1) == num)[0]
            
            # Remove 0 columns
            # Removing 0 columns cannot be done all in once, as not all 0 columns are in the same index
            B = np.array([self.P[state_idx, :, possible_transitions[state_idx]] for state_idx in states])
            # If an element of B is zero, its entire column must be 0, otherwise, replace the problematic element by epsilon and renormalize
            B[B == 0] = epsilon
            B /= np.sum(B, axis=1, keepdims=True)
            

            log_B = np.where(B != 0, np.log(B), B)
            # y = self.R[state] + lmdp_tdr.lmbda * np.sum(B * log_B, axis=1)
            y = self.R[states] - self.temperature * np.log(1 / self.policy_ref[states, :]) + lmdp_tdr.lmbda * np.sum(B * log_B, axis=1)
            B_dagger = np.linalg.pinv(B.astype(np.float64))
            
            x = np.einsum("sap,sp->sa", B_dagger.transpose(0, 2, 1), y)

            next_states = np.where(possible_transitions[states])[1]
            states = np.repeat(states, num)
            lmdp_tdr.R[states, next_states] = (x + lmdp_tdr.lmbda * np.log(num)).flatten()
            
            # assert all(lmdp_tdr.R[state, ~zero_cols] <= 0), f"Not all rewards for origin state {state} are negative:\n{lmdp_tdr.R[state, ~zero_cols]}"
            lmdp_tdr.P[states, next_states] = np.exp(-np.log(num))
                
        lmdp_tdr.R = csr_matrix(lmdp_tdr.R)
        lmdp_tdr.P = csr_matrix(lmdp_tdr.P)
        
        stats.end_time()
        
        z, _, overflow = lmdp_tdr.power_iteration()
        
        if not overflow:
            V_lmdp = lmdp_tdr.get_value_function(z)
            if not hasattr(self, "V"):
                self.V, self.stats, _ = self.value_iteration()
            
            self._print(f"EMBEDDING ERROR: {np.mean(np.square(V_lmdp - self.V))}")
        
        return lmdp_tdr, stats, overflow
    
    
    def to_LMDP_TDR(self, lmbda: float = None, find_best_lmbda: bool = True, vectorized: bool = True) -> tuple[LMDP_TDR, EmbeddingStats, bool]:
        """
        A wrapper function to convert the MDP into an LMDP with transition-dependent rewards.
        It chooses between the vectorized and iterative implementations based on the `vectorized` parameter.
        
        Args:
            lmbda (float, optional): The temperature parameter for the LMDP. If None, defaults to 1. Defaults to None.
            find_best_lmbda (bool): If True, finds the best lambda value for the LMDP embedding. Defaults to True.
            vectorized (bool): If True, uses the vectorized implementation; otherwise, uses the iterative one. Defaults to True.
        
        Returns:
            tuple[LMDP_TDR, EmbeddingStats, bool]: A tuple containing:
                - lmdp_tdr (LMDP_TDR): The LMDP-TDR representation of the MDP.
                - stats (EmbeddingStats): Statistics about the embedding process.
                - overflow (bool): A boolean indicating whether an overflow occurred during the computation.
        """
        if vectorized:
            return self.to_LMDP_TDR_vectorized(lmbda, find_best_lmbda)
        else:
            return self.to_LMDP_TDR_iterative(lmbda, find_best_lmbda)
    

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

    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)
    
    
    def __eq__(self, obj):
        """
        Compare two MDP objects to check if they are equal.
        The equality is defined as having the same attributes except for the excluded ones.
        Args:
            obj (MDP): The MDP object to compare.
        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(obj, MDP):
            return False

        exclude_attributes = ["verbose", "num_actions", "__allowed_actions"] # Num actions and __allowed_actions can be ommitted from the comparison because they are accounted in the matrix P.
        return compare_models(self, obj, exclude_attributes=exclude_attributes)

    
    """
    SOME ALTERNATIVE EMBEDDING METHODS TO CONVERT AN MDP INTO AN LMDP WITH TRANSITION-DEPENDENT REWARDS THAT HAVE NOT WORKED AS EXPECTED
    """
    # def to_LMDP_TDR_2(self):
    #     self._print(f"Computing the LMDP-TDR embedding of this MDP...")
        
    #     lmdp_tdr = LMDP_TDR(
    #         num_states=self.num_states,
    #         num_terminal_states=self.num_terminal_states,
    #         sparse_optimization=True,
    #         lmbda=1,
    #         s0=self.s0,
    #         verbose=self.verbose,
    #         dtype=self.dtype
    #     )
        
    #     R_1 = np.einsum("san,na->sn", self.P, self.R) / self.num_actions
    #     R_1[R_1 == 0] = 1e-10
                
    #     epsilon = 1e-10
    #     for state in range(self.num_non_terminal_states):
    #         # print(f"STATE: {state}")
    #         B = self.P[state, :, :]
    #         zero_cols = np.all(B == 0, axis=0)
    #         zero_cols_idx = np.where(zero_cols)[0]
            
    #         # Remove 0 columns
    #         B = B[:, ~zero_cols]
    #         curr_r1 = R_1[state, ~zero_cols]
            
    #         # If an element of B is zero, its entire column must be 0, otherwise, replace the problematic element by epsilon and renormalize
    #         B[B == 0] = epsilon
    #         B /= np.sum(B, axis=1).reshape(-1, 1)
            

    #         log_B = np.where(B != 0, np.log(B), B)
    #         y_1 = self.R[state] + np.sum(B * log_B, axis=1)
    #         y = self.R[state] + np.sum(B * (lmdp_tdr.lmbda * log_B - curr_r1), axis=1)
            
    #         test_1 = np.zeros_like(self.R[state])
    #         for s in range(self.num_states):
    #             if s in zero_cols_idx: continue
    #             test_1 += self.P[state, :, s] * (np.log(self.P[state, :, s]))
            
    #         test_1 += self.R[state]
    #         assert np.all(test_1 == y_1)
            
    #         test_2 = np.zeros_like(self.R[state])
    #         for s in range(self.num_states):
    #             if s in zero_cols_idx: continue
    #             test_2 += self.P[state, :, s] * (np.log(self.P[state, :, s]) - R_1[state, s])
            
    #         test_2 += self.R[state]
            
    #         assert np.all(test_2 == y)
            
    #         B_dagger = np.linalg.pinv(B)
    #         x_1 = B_dagger @ y_1 # From first version of the embedding to do some checkings
    #         x = B_dagger @ y
            
    #         assert np.all(np.isclose(np.sum(B_dagger * test_2, axis=1) + curr_r1, x_1)) # If this does not fail, it means that the results obtained here are the same as in the first embedding version and therefore, the LMDP and LMDP with TDR will be equivalent
            
    #         support_x = [col for col in zero_cols if col == False]
    #         len_support = len(support_x)
            
            
    #         lmdp_tdr.R[state, ~zero_cols] = x + lmdp_tdr.lmbda * np.log(len_support) # R_2
    #         lmdp_tdr.P[state, ~zero_cols] = np.exp(-np.log(len_support))
                
    #     lmdp_tdr.R += R_1
    #     lmdp_tdr.R = csr_matrix(lmdp_tdr.R)
    #     lmdp_tdr.P = csr_matrix(lmdp_tdr.P)
        
    #     z, _, overflow = lmdp_tdr.power_iteration()
    #     if not overflow:
    #         V_lmdp = lmdp_tdr.get_value_function(z)
    #         V_mdp, _, _ = self.value_iteration()
            
    #         self._print("EMBEDDING ERROR:", np.mean(np.square(V_lmdp - V_mdp)))    
        
    #     return lmdp_tdr, overflow

    
    # def to_LMDP_TDR_3(self):
    #     self._print(f"Computing the LMDP-TDR embedding of this MDP...")
        
    #     self.compute_value_function()
    #     lmdp_tdr = LMDP_TDR(
    #         num_states=self.num_states,
    #         num_terminal_states=self.num_terminal_states,
    #         sparse_optimization=False,
    #         lmbda=1,
    #         s0=self.s0,
    #         verbose=self.verbose,
    #         dtype=self.dtype
    #     )
        
    #     epsilon = 1e-10
    #     for state in range(self.num_non_terminal_states):
    #         x = np.sum(self.P[state, :, :] * self.R[state, :].reshape(-1, 1), axis=0)
    #         denominator = np.sum(self.P[state, :, :], axis=0)
    #         nonzero_cols = np.where(x != 0)[0]
    #         len_support = len(nonzero_cols)
            
    #         x = np.where(denominator != 0, x / denominator, 0)
            
    #         if state == 0: self._print("x ldmp-tdr", x)
            
    #         # lmdp_tdr.P[state, nonzero_cols] = np.exp(-np.log(len_support))
    #         lmdp_tdr.P[state] = self.P[state, self.policy[state]]
    #         lmdp_tdr.R[state] = x + lmdp_tdr.lmbda * np.log(len_support)
        
        
    #     z, _, overflow = lmdp_tdr.power_iteration()
        
    #     if not overflow:
    #         V_lmdp = lmdp_tdr.get_value_function(z)
    #         V_mdp, _, _ = self.value_iteration()
            
    #         self._print("EMBEDDING ERROR MDP to LMDP-TDR:", np.mean(np.square(V_lmdp - V_mdp)))    
        
    #     return lmdp_tdr, overflow