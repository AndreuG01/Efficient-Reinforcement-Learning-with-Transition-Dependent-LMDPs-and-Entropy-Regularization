import numpy as np
from scipy.sparse import csr_matrix
from collections.abc import Callable
from domains.grid import CustomGrid
from sys import getsizeof
from joblib import Parallel, delayed, cpu_count
import time
from tqdm import tqdm
import models
import models.LMDP
from utils.coloring import TerminalColor
from utils.utils import print_overflow_message
from .utils import compare_models
from utils.stats import ModelBasedAlgsStats

class LMDP_TDR:
    """
    A class representing a Linearly-Solvable Markov Decision Process (LMDP) with Transition-Dependent Rewards (TDR)
    
    The LMDP-TDR is defined by a 4-tuple: (S, P, R, lambda) where:
        - S: A finite set of states (num_states)
        - P: A state transition probability function P(s'| s). Known as passive dynamics.
        - R: A reward function R(s, s')
        - lambda: A temperature parameter that controls the penalty for the LMDP deviating from the passive dynamics.
        
    Attributes:
        num_states (int): Total number of states in the LMDP.
        num_terminal_states (int): Number of terminal states in the LMDP.
        num_non_terminal_states (int): Number of non-terminal states in the LMDP.
        s0 (int): The initial state index.
        lmbda (int): The temperature parameter for the LMDP.
        sparse_optimization (bool): Whether to use sparse optimization for the transition matrix P and reward matrix R.
        verbose (bool): Whether to print verbose output during operations.
        dtype (np.dtype): The data type for the matrices, can be np.float32, np.float64, or np.float128.
        P (np.ndarray | csr_matrix): The transition probability matrix of the LMDP.
        R (np.ndarray): The reward matrix of the LMDP.
    """
    def __init__(
        self,
        num_states: int,
        num_terminal_states: int,
        lmbda: int = 1,
        s0: int = 0,
        sparse_optimization: bool = True,
        verbose: bool = True,
        dtype: np.dtype = np.float128
    ) -> None:
        """
        Initializes the LMDP_TDR instance with the given parameters.
        
        Args:
            num_states (int): Total number of states in the LMDP.
            num_terminal_states (int): Number of terminal states in the LMDP.
            lmbda (int): The temperature parameter for the LMDP. Defaults to 1.
            s0 (int): The initial state index. Defaults to 0.
            sparse_optimization (bool): Whether to use sparse optimization for the transition matrix P and reward matrix R. Defaults to True.
            verbose (bool): Whether to print verbose output during operations. Defaults to True.
            dtype (np.dtype): The data type for the matrices, can be np.float32, np.float64, or np.float128. Defaults to np.float128.
        """
        
        assert dtype in [np.float32, np.float64, np.float128], f"Only allowed data types: {[np.float32, np.float64, np.float128]}"
        self.dtype = dtype
        
        self.num_states = num_states
        self.num_terminal_states = num_terminal_states
        self.num_non_terminal_states = self.num_states - self.num_terminal_states
        self.s0 = s0
        self.lmbda = lmbda
        self.sparse_optimization = sparse_optimization
        
        self.P: np.ndarray | csr_matrix = np.zeros((self.num_non_terminal_states, self.num_states), dtype=self.dtype)
        self.R = np.zeros((self.num_non_terminal_states, self.num_states), dtype=self.dtype)
        
        self.verbose = verbose
    
    
    def generate_P(self, grid: CustomGrid, actions: list[int], num_threads: int = 10, benchmark: bool = False) -> float:
        """
        Generates the transition probability matrix (P) for the LMDP, based on the dynamics of the environment.

        Args:
            grid (CustomGrid): The grid environment containing the states and their properties, as well as the movement logic.
            actions (list[int]): List of actions that can be taken in the environment.
            num_threads (int): Number of threads to use for parallel processing. Defaults to 10.
            benchmark (bool): If True, measures the time taken to generate the transition matrix. Defaults to False.
        
        Returns:
            float: The total time taken to generate the transition matrix if benchmark is True, otherwise 0.
        """
        pos = grid.states
        terminal_pos = grid.terminal_states
        
        def process_state(state: int) -> list[float]:
            row_updates = []
            for action in actions:
                if grid.is_cliff(grid.state_index_mapper[state]):
                    next_state = self.s0
                else:
                    next_state, _, terminal = grid.move(pos[state], action)
                    # Convert from coordinate-like system (i, j) (grid format) to index based (idx) (matrix format)
                    if terminal:
                        next_state = len(pos) + terminal_pos.index(next_state)
                    else:
                        next_state = pos.index(next_state)
                
                row_updates.append((state, next_state, 1 / len(actions)))
            
            return row_updates
        
        total_time = 0
        if benchmark:
            start_time = time.time()
        results = Parallel(n_jobs=min(num_threads, cpu_count()), temp_folder="/tmp")(
            delayed(process_state)(state) for state in tqdm(range(self.num_non_terminal_states), 
                                                         desc="Generating transition matrix P", 
                                                         total=self.num_non_terminal_states,
                                                         disable=not self.verbose)
        )
        

        for row_updates in results:
            for state_idx, next_state_idx, prob in row_updates:
                self.P[state_idx, next_state_idx] += prob
        
        if benchmark:
            end_time = time.time()
            total_time = end_time - start_time
                
    
        assert all([np.isclose(np.sum(self.P[i, :]), 1) for i in range(self.P.shape[0])]), "Transition probabilities are not properly defined. They do not add up to 1 in every row"
        
        
        self._print(f"Generated matrix P with {self.P.size:,} elements")
        if self.sparse_optimization:
            self._print("Converting P into sparse matrix...")
            self._print(f"Memory usage before conversion: {getsizeof(self.P):,} bytes")
            self.P = csr_matrix(self.P)
            self._print(f"Memory usage after conversion: {getsizeof(self.P):,} bytes")
        
        return total_time
    
    
    def _generate_R(self):
        raise NotImplementedError("Implement in the subclass.")
    
    
    def transition(self, state: int) -> tuple[int, float, bool]:
        """
        Simulates a transition from the given state to the next state based on the transition probabilities.
        
        Args:
            state (int): The current state index from which to transition.
        
        Returns:
            tuple[int, float, bool]: A tuple containing:
                - next_state (int): The index of the next state after the transition.
                - reward (float): The reward received for transitioning to the next state.
                - is_terminal (bool): A boolean indicating whether the next state is a terminal state.
        """
        next_state = np.random.choice(self.num_states, p=self.P[state] if self.dtype != np.float128 else self.P[state].astype(np.float64))
        
        return (
            next_state,
            self.R[next_state],
            next_state >= self.num_non_terminal_states
        )
    
    
    def get_control(self, z: np.ndarray, o: np.ndarray | csr_matrix) ->np.ndarray:
        """
        Computes the control vector (or policy) for the LMDP-TDR.
        
        Args:
            z (np.ndarray): The exponentiated value function vector for the LMDP-TDR.
            o (np.ndarray | csr_matrix): The reward matrix.
        
        Returns:
            np.ndarray: The control vector (or policy) for the LMDP-TDR, normalized to sum to 1 across each row.
        """
        
        if self.sparse_optimization:
            control = self.P.multiply(o).multiply(z)
        else:
            control = self.P * o * z
        
        control = control.astype(self.dtype)
        control = control / np.sum(control, axis=1).reshape(-1, 1)
        # print(f"Control elements: {control.size}. Non zero: {np.count_nonzero(control)}")
        assert all(np.isclose(np.sum(control, axis=1), 1))
        
        return control
        
    
    def get_optimal_policy(self, z: np.ndarray, o: np.ndarray | csr_matrix) -> np.ndarray:
        """
        Computes the optimal policy for the LMDP-TDR based on the value function and reward matrix. It is a wrapper around `get_control`.
        
        Args:
            z (np.ndarray): The exponentiated value function vector for the LMDP-TDR.
            o (np.ndarray | csr_matrix): The reward matrix.
        
        Returns:
            np.ndarray: The optimal policy for the LMDP-TDR, where each row corresponds to a state and contains the probabilities of transitioning to the next states.
        """
        probs = self.get_control(z, o)
        if self.sparse_optimization:
            probs = probs.toarray()
        return probs

    
    def power_iteration(self, epsilon=1e-10, temp: float = None) -> tuple[np.ndarray, ModelBasedAlgsStats, bool]:
        """
        Performs power iteration to compute the value function for the LMDP-TDR.
        The algorithm iteratively computes the exponentiated value function vector until convergence using the following formula:
                                                    z = Gz^+
                                                    z^+ = z || reward for the terminal states 
        
        Args:
            epsilon (float): The convergence threshold for the power iteration. Defaults to 1e-10.
            temp (float): The temperature parameter for the LMDP-TDR. If None, uses the instance's lmbda attribute. Defaults to None.
        
        Returns:
            tuple[np.ndarray, ModelBasedAlgsStats, bool]: A tuple containing:
                - z (np.ndarray): The exponentiated value function vector for the LMDP-TDR.
                - stats (ModelBasedAlgsStats): Statistics about the power iteration process, including time taken and number of iterations.
                - overflow (bool): A boolean indicating whether an overflow occurred during the computation.
        """
        if temp is not None:
            temperature = temp
        else:
            temperature = self.lmbda
        
        self._print(f"Power iteration LMDP-TDR...")
        if self.sparse_optimization:
            self.o = np.exp(self.R.toarray() / temperature) # TODO: I have problems when this is a sparse matrix
            G = csr_matrix(self.P.multiply(self.o))
            
        else:
            self.o = np.exp(self.R / temperature)
            G = self.P * self.o
    
    
        z = np.ones(self.num_states, dtype=self.dtype) * 3
        iterations = 0
        delta = 0
        overflow = False
        
        start_time = time.time()
        Vs = []
        deltas = []
        
        while True:
            z_new = G @ z
            z_new = np.concatenate((z_new, np.ones((self.num_terminal_states), dtype=self.dtype)))
            
            if np.inf in z_new or -np.inf in z_new:
                overflow = True
                print_overflow_message(z_new, z, self.dtype, temperature)
                break
            
            Vs.append(self.get_value_function(z_new, temp=temp))
            delta = np.linalg.norm(self.get_value_function(z_new, temp=temp) - self.get_value_function(z, temp=temp), ord=np.inf)
            deltas.append(delta)
            
            if iterations % 1000 == 0:
                self._print(f"Iter: {iterations}. Delta: {delta}")
        
            if delta < epsilon:
                break
            
            z = z_new
            iterations += 1
        
        elapsed_time = time.time() - start_time
        
        self.z = z
        self._print(f"Converged in {iterations} iterations")
        return z, ModelBasedAlgsStats(elapsed_time, iterations, deltas, self.num_states, Vs, "PI"), overflow
        
    
    def get_value_function(self, z: np.ndarray = None, temp: float = None) -> np.ndarray:
        """
        Computes the value function for the LMDP-TDR based on the exponentiated value function vector z.
        Since z = e^(V / lambda), to recover the original value function, the opposite transformation is done: V = lambda * log(z).
        
        Args:
            z (np.ndarray): The exponentiated value function vector for the LMDP-TDR. If None, uses the instance's z attribute. Defaults to None.
            temp (float): The temperature parameter for the LMDP-TDR. If None, uses the instance's lmbda attribute. Defaults to None.
            
        Returns:
            np.ndarray: The value function for the LMDP-TDR, where each element corresponds to the value of a state.
        """
        if temp is not None:
            temperature = temp
        else:
            temperature = self.lmbda
        
        if z is None:
            z = self.z
        
        result = np.log(z, dtype=self.dtype) * temperature
        result[result == -np.inf] = np.finfo(self.dtype).min
        
        return result
    
    
    def compute_value_function(self, temp: float = None) -> None:
        """
        Computes the value function for the LMDP-TDR using power iteration.
        This method calls the `power_iteration` method to compute the exponentiated value function vector z,
        and then computes the value function V using the `get_value_function` method.
        
        Args:
            temp (float): The temperature parameter for the LMDP-TDR. If None, uses the instance's lmbda attribute. Defaults to None.
        
        Returns:
            None
        """
        # if not hasattr(self, "z"):
        self._print("Will compute power iteration")
        _, self.stats, _ = self.power_iteration(temp=temp)
        
        self.V = self.get_value_function(temp=temp)
        
        self.policy = self.get_optimal_policy(self.z, self.o)
        
    
    def to_MDP(self):
        raise NotImplementedError("Not implemented yet")
    
    
    def to_LMDP(self) -> models.LMDP:
        """
        Converts the LMDP with transition dependent rewards to an equivalent LMDP with state-dependent rewards, using the transformations
        derived in this thesis.
        
        First:
                                        x_s(s') = R(s, s') + lambda * log(P(s' | s))
        
        Then, the reward function is defined as:
                                R(s) = lambda * log(sum(exp(x_s(s') / lambda))) for all s in S
        
        The transition probabilities are defined as:
                            P(s' | s) = exp(x_s(s') / lambda) / sum(exp(x_s(s') / lambda)) for all s in S
        
        Returns:
            models.LMDP: the equivalent LMDP with state-dependent rewards.
        """
        lmdp = models.LMDP(
            self.num_states,
            self.num_terminal_states,
            self.lmbda,
            self.s0,
            sparse_optimization=False,
            verbose=self.verbose,
            dtype=self.dtype
        )
        
        large_negative = self.dtype(-1e10)
        x = self.R + self.lmbda * np.where(self.P != 0, np.log(self.P), large_negative)
        
        lmdp.R = self.lmbda * np.log(np.sum(np.exp(x / self.lmbda), axis=1))
        # Add the rewards for the terminal states
        lmdp.R = np.append(lmdp.R, np.zeros(self.num_terminal_states))
        
        lmdp.P = np.exp(x / self.lmbda)
        lmdp.P /= np.sum(lmdp.P, axis=1).reshape(-1, 1)

        return lmdp
    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)
    
    def __eq__(self, obj):
        """
        Compare two LMDP_TDR objects to check if they are equal.
        The equality is defined as having the same attributes except for the excluded ones.
        Args:
            obj (LMDP_TDR): The LMDP_TDR object to compare.
        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        exclude_attributes = ["verbose", "sparse_optimization"]
        return compare_models(self, obj, exclude_attributes=exclude_attributes)