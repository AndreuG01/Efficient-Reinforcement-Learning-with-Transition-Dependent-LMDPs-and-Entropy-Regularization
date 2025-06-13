import numpy as np
from collections.abc import Callable
from domains.grid import CustomGrid
import models
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sys import getsizeof
from joblib import Parallel, delayed, cpu_count
import time
import models.MDP
import models.LMDP_TDR
from utils.stats import ModelBasedAlgsStats
from utils.utils import print_overflow_message
from .utils import compare_models

class LMDP:
    """
    A class representing a Linearly-Solvable Markov Decision Process (LMDP) as described by Todorov, 2006.
    
    The LMDP is defined by a 4-tuple: (S, P, R, lambda) where:
        - S: A finite set of states (num_states)
        - P: A state transition probability function P(s'| s)
        - R: A reward function R(s)
        - lambda: A temperature parameter that controls the penalty for the LMDP deviating from the passive dynamics.
    
    Attributes:
        num_states (int): The total number of states in the MDP.
        num_terminal_states (int): The number of terminal states.
        num_non_terminal_states (int): The number of non-terminal states.
        s0 (int): The initial state index.
        lmbda (int): The temperature parameter for the LMDP.
        sparse_optimization (bool): Whether to use sparse matrix optimization for the transition matrix.
        verbose (bool): Whether to print verbose output during computations.
        P (np.ndarray | csr_matrix): The state transition probability matrix.
        R (np.ndarray): The reward matrix for each state-action pair.
        dtype (np.dtype): The data type for the matrices, can be np.float32, np.float64, or np.float128.
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
        Initialize the LMDP with the given parameters.
        
        Args:
            num_states (int): Total number of states in the LMDP.
            num_terminal_states (int): Number of terminal states in the LMDP.
            lmbda (int): Regularization factor for KL divergence (default is 1).
            s0 (int): Initial state index (default is 1).
            sparse_optimization (bool): Whether to use sparse matrix optimization for the transition matrix (default is True).
            verbose (bool): Whether to print verbose output during computations (default is True).
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
        self.R = np.zeros(self.num_states, dtype=self.dtype)
        
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
        
    
    def get_control(self, z: np.ndarray) ->np.ndarray:
        """
        Computes the control vector (or policy) for the LMDP.
        
        Args:
            z (np.ndarray): The exponentiated value function vector for the LMDP-TDR.
        
        Returns:
            np.ndarray: The control vector (or policy) for the LMDP-TDR, normalized to sum to 1 across each row.
        """
        if type(self.P) == csr_matrix:
            # TODO: keep working with sparse matrices here.
            self.P = self.P.toarray()
        
        control = (self.P * z).astype(self.dtype)
        control = control / np.sum(control, axis=1).reshape(-1, 1)
        
        # print(f"Control elements: {control.size}. Non zero: {np.count_nonzero(control)}")
        # assert all(np.isclose(np.sum(control, axis=1), 1)), np.sum(control, axis=1)
        
        return control
        
    
    def get_optimal_policy(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the optimal policy for the LMDP based on the exponentiated value function. It is a wrapper around `get_control`.
        
        Args:
            z (np.ndarray): The exponentiated value function vector for the LMDP.
        
        Returns:
            np.ndarray: The optimal policy for the LMDP, where each row corresponds to a state and contains the probabilities of transitioning to the next states.
        """
        return self.get_control(z)
        
    
    
    def power_iteration(self, epsilon=1e-10, max_iterations=100000) -> tuple[np.ndarray, ModelBasedAlgsStats, bool]:
        """
        Performs power iteration to compute the value function for the LMDP.
        The algorithm iteratively computes the exponentiated value function vector until convergence using the following formula:
                                                    z = GPz^+
                                                    z^+ = z || reward for the terminal states 
        
        Args:
            epsilon (float): The convergence threshold for the power iteration. Defaults to 1e-10.
            max_iterations (int): The maximum number of iterations to perform. Defaults to 100000.
        
        Returns:
            tuple[np.ndarray, ModelBasedAlgsStats, bool]: A tuple containing:
                - z (np.ndarray): The exponentiated value function vector for the LMDP-TDR.
                - stats (ModelBasedAlgsStats): Statistics about the power iteration process, including time taken and number of iterations.
                - overflow (bool): A boolean indicating whether an overflow occurred during the computation.
        """
        G = np.diag(np.exp(self.R[:self.num_non_terminal_states] / self.lmbda))
        z = np.ones(self.num_states, dtype=self.dtype)
        
        self._print(f"Power iteration LMDP...")
        if self.sparse_optimization:
            if type(self.P) != csr_matrix: self.P = csr_matrix(self.P)
            G = csr_matrix(G)
                
        iterations = 0
        start_time = time.time()
        deltas = []
        Vs = []
        overflow = False
        
        while True:
            delta = 0
            z_new = G @ self.P @ z
            z_new = np.concatenate((z_new, np.ones((self.num_terminal_states), dtype=self.dtype)), dtype=self.dtype)
            
            if np.inf in z_new or -np.inf in z_new:
                overflow = True
                print_overflow_message(z_new, z, self.dtype, self.lmbda)
                break
            
            Vs.append(self.get_value_function(z_new))
            
            delta = np.linalg.norm(self.get_value_function(z_new) - self.get_value_function(z), ord=np.inf)
            
            if iterations % 100 == 0:
                self._print(f"Iter: {iterations}. Delta: {delta}")

            if delta < epsilon or iterations == max_iterations:
                break

            z = z_new
            iterations += 1
            deltas.append(delta)
        
        elapsed_time = time.time() - start_time
        
        self.z = z
        self._print(f"Converged in {iterations} iterations")
        return z, ModelBasedAlgsStats(elapsed_time, iterations, deltas, self.num_states, Vs, "PI"), overflow

    def get_value_function(self, z: np.ndarray = None) -> np.ndarray:
        """
        Computes the value function for the LMDP based on the exponentiated value function vector z.
        Since z = e^(V / lambda), to recover the original value function, the opposite transformation is done: V = lambda * log(z).
        
        Args:
            z (np.ndarray): The exponentiated value function vector for the LMDP-TDR. If None, uses the instance's z attribute. Defaults to None.
            
        Returns:
            np.ndarray: The value function for the LMDP, where each element corresponds to the value of a state.
        """
        if z is None:
            z = self.z
        
        result = np.log(z, dtype=self.dtype) * self.lmbda
        result[result == -np.inf] = np.finfo(self.dtype).min
        
        return result
    
    
    def compute_value_function(self) -> None:
        """
        Computes the value function for the LMDP using power iteration.
        This method calls the `power_iteration` method to compute the exponentiated value function vector z,
        and then computes the value function V using the `get_value_function` method.
        
        Args:
            None
        
        Returns:
            None
        """
        if not hasattr(self, "z"):
            self._print("Will compute power iteration")
        _, self.stats, _ = self.power_iteration()
        
        self.V = self.get_value_function()
        
        self.policy = self.get_optimal_policy(self.z)
        
    
    def to_MDP(self):
        """
        Convert the LMDP to an equivalent MDP.

        Returns:
            mdp (models.MDP): The equivalent Markov Decision Process.
        """
        z, _, _ = self.power_iteration()
        
        control = self.get_control(z)
        self._print(f"Computing the MDP embedding of this LMDP...")
        # The minimum number of actions that can be done to achieve the same behavior in an MDP.
        num_actions = np.max(np.sum(control > 0, axis=1))
        
        mdp = models.MDP(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            allowed_actions=[i for i in range(num_actions)],
            s0=self.s0,
            verbose=self.verbose,
            dtype=self.dtype
            # gamma=0.9
        )
        
        # Define the reward function of the MDP.
        kl_term = np.zeros_like(control)
        mask = control > 0
        kl_term[mask] = control[mask] * np.log(control[mask] / self.P[mask])
        kl_term = np.sum(kl_term, axis=1)
        
        reward_non_terminal = self.R[:self.num_non_terminal_states] - self.lmbda * kl_term
        reward_non_terminal = np.tile(reward_non_terminal.reshape(-1, 1), (1, num_actions))
        
        mdp.R[:self.num_non_terminal_states, :] = reward_non_terminal
        
        # Define the transition probability matrix of the MDP.
        for s in tqdm(range(self.num_non_terminal_states), desc="Generating transition matrix P", total=self.num_non_terminal_states, disable=not self.verbose):
            non_zero_positions = np.where(control[s, :] != 0)[0]
            probs = control[s, non_zero_positions].flatten()
            for i, a in enumerate(range(num_actions), start=1):
                mdp.P[s, a, non_zero_positions] = probs
                probs = np.roll(probs, shift=1, axis=0)
        
        V_lmdp = self.get_value_function(z)
        
        mdp.compute_value_function()
        V_mdp = mdp.V
        
        self._print(f"EMBEDDING ERROR: {np.mean(np.square(V_lmdp - V_mdp))}")
        
        return mdp


    def to_LMDP_TDR(self):
        """
        Converts the LMDP with state-dependent rewards to an equivalent LMDP with transition-dependent rewards (LMDP-TDR).
        The transition probability matrix P remains the same, but the reward function is transformed to account for the transition-dependent nature of the rewards.
                                                
                                                R(s,s') = R(s) for all s in S, s' in S.
                                                
        Args:
            None
        
        Returns:
            models.LMDP_TDR: The equivalent LMDP with transition-dependent rewards.
        """
        lmdp_tdr = models.LMDP_TDR(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            s0=self.s0,
            sparse_optimization=self.sparse_optimization,
            verbose=self.verbose,
            dtype=self.dtype
        )
        
        lmdp_tdr.P = self.P.copy()
        for state in range(self.num_non_terminal_states):
            lmdp_tdr.R[state, :] = np.full(shape=self.num_states, fill_value=self.R[state], dtype=self.dtype)
        
        if self.sparse_optimization:
            lmdp_tdr.R = csr_matrix(lmdp_tdr.R)
        
        return lmdp_tdr
    
    def _print(self, msg, end: str = "\n"):
        if self.verbose:
            print(msg, end=end)
            
    def __eq__(self, obj):
        """
        Compare two LMDP objects to check if they are equal.
        The equality is defined as having the same attributes except for the excluded ones.
        Args:
            obj (LMDP): The LMDP object to compare.
        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(obj, LMDP):
            return False

        exclude_attributes = ["verbose", "sparse_optimization"] # Num actions and __allowed_actions can be ommitted from the comparison because they are accounted in the matrix P.
        return compare_models(self, obj, exclude_attributes=exclude_attributes)