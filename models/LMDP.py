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
from utils.stats import ModelBasedAlgsStats

class LMDP:
    """
    A class representing a Linear Markov Decision Process (LMDP).
    
    The LMDP is defined by a 3-tuple: (S, P, R) where:
    - S: A finite set of states (num_states)
    - P: A state transition probability function P(s'| s)
    - R: A reward function R(s)
    
    Attributes:
    - num_states (int): The total number of states in the MDP.
    - num_terminal_states (int): The number of terminal states.
    - num_non_terminal_states (int): The number of non-terminal states.
    - s0 (int): The initial state index.
    - gamma (float): The temperature parameter
    - P (np.ndarray): The state transition probability matrix.
    - R (np.ndarray): The reward matrix for each state-action pair.
    # TODO: complete when code is finished
    """
    def __init__(self, num_states: int, num_terminal_states: int, lmbda: int = 1, s0: int = 0, sparse_optimization: bool = True) -> None:
        """
        Initialize the LMDP with the given parameters.
        
        Args:
        - num_states (int): Total number of states in the LMDP.
        - num_terminal_states (int): Number of terminal states in the LMDP.
        - lmbda (int): Regularization factor for KL divergence (default is 1).
        - s0 (int): Initial state index (default is 1).
        """
        self.num_states = num_states
        self.num_terminal_states = num_terminal_states
        self.num_non_terminal_states = self.num_states - self.num_terminal_states
        self.s0 = s0
        self.lmbda = lmbda
        self.sparse_optimization = sparse_optimization
        
        self.P: np.ndarray | csr_matrix = np.zeros((self.num_non_terminal_states, self.num_states))
        self.R = np.zeros(self.num_states, dtype=np.float64)
        
    
    
    def generate_P(self, move: Callable, grid: CustomGrid, actions: list[int], num_threads: int = 10, benchmark: bool = False) -> float:
        """
        Generates the transition probability matrix (P) for the LMDP, based on the dynamics of the environment.

        Args:
        - pos (dict[int, list]): The different positions of the grid
        - move (Callable): A function that determines the next state based on the current state and action.
        The function signature should be `move(state: State, action: int) -> tuple[next_state: State, reward: float, done: bool]`.
        - grid (CustomGrid): The grid environment for which the transition matrix is being generated.
        - actions (list[int]): List of possible actions that can be taken in the environment (NOTE THAT THE LMDP DOES NOT HAVE ACTIONS INTO ACOUNT)
        """
        pos = grid.states
        terminal_pos = grid.terminal_states
        
        def process_state(state: int) -> list[float]:
            row_updates = []
            for action in actions:
                if grid.is_cliff(grid.state_index_mapper[state]):
                    next_state = self.s0
                else:
                    next_state, _, terminal = move(pos[state], action)
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
        results = Parallel(n_jobs=min(num_threads, cpu_count()))(
            delayed(process_state)(state) for state in tqdm(range(self.num_non_terminal_states), 
                                                         desc="Generating transition matrix P", 
                                                         total=self.num_non_terminal_states)
        )
        

        for row_updates in results:
            for state_idx, next_state_idx, prob in row_updates:
                self.P[state_idx, next_state_idx] += prob
        
        if benchmark:
            end_time = time.time()
            total_time = end_time - start_time
                
    
        assert all([np.isclose(np.sum(self.P[i, :]), 1) for i in range(self.P.shape[0])]), "Transition probabilities are not properly defined. They do not add up to 1 in every row"
        
        
        print(f"Generated matrix P with {self.P.size:,} elements")
        if self.sparse_optimization:
            print("Converting P into sparse matrix...")
            print(f"Memory usage before conversion: {getsizeof(self.P):,} bytes")
            self.P = csr_matrix(self.P)
            print(f"Memory usage after conversion: {getsizeof(self.P):,} bytes")
        
        return total_time
    
    
    def _generate_R(self):
        raise NotImplementedError("Implement in the subclass.")
    
    
    def transition(self, state: int) -> tuple[int, float, bool]:
        """
        Simulate a state transition given the current state.

        Args:
        - state (int): The current state.

        Returns:
        - next_state (int): The state reached after the transition.
        - reward (float): The reward obtained for the transition.
        - terminal (bool): True if the next state is a terminal state, False otherwise.
        """
        next_state = np.random.choice(self.num_states, p=self.P[state])
        
        return (
            next_state,
            self.R[next_state],
            next_state >= self.num_non_terminal_states
        )
        
    
    def get_control(self, z: np.ndarray) ->np.ndarray:
        """
        Compute the controlled transition probability matrix based on the value function approximation.

        Args:
        - z (np.ndarray): The transfomed value function vector.

        Returns:
        - control (np.ndarray): The controlled transition probability matrix
        """
        if type(self.P) == csr_matrix:
            # TODO: keep working with sparse matrices here.
            self.P = self.P.toarray()
        
        control = self.P * z
        control = control / np.sum(control, axis=1).reshape(-1, 1)
        
        # print(f"Control elements: {control.size}. Non zero: {np.count_nonzero(control)}")
        assert all(np.isclose(np.sum(control, axis=1), 1))
        
        return control
        
    
    def get_optimal_policy(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the optimal policy based on the control matrix.

        Args:
        - z (np.ndarray): The transfomed value function vector.

        Returns:
        - policy (np.ndarray): The optimal policy.
        """
        return self.get_control(z)
        
    
    def transition_action(self, state: int, next_state: list[int]) -> list[int]:
        # LMDPs do not have actions. However, to be able to plot the policies, or interact with the environment, we need to convert the transitions into certain actions
        # (as long as the problem is deterministic)
        raise NotImplementedError("Implement in the subclass")
    
    
    def power_iteration(self, epsilon=1e-10) -> np.ndarray:
        """
        Perform power iteration to compute the value function approximation.

        Args:
        - epsilon (float): Convergence threshold (default is 1e-20).

        Returns:
        - z (np.ndarray): Converged transformed value function vector.
        """
        G = np.diag(np.exp(self.R[:self.num_non_terminal_states] / self.lmbda))
        z = np.ones(self.num_states)
        
        print(f"Power iteration...")
        if self.sparse_optimization:
            if type(self.P) != csr_matrix: self.P = csr_matrix(self.P)
            G = csr_matrix(G)
                
        iterations = 0
        start_time = time.time()
        deltas = []
        Vs = []
        
        while True:
            delta = 0
            z_new = G @ self.P @ z
            z_new = np.concatenate((z_new, np.ones((self.num_terminal_states))))
            Vs.append(self.get_value_function(z_new))
            
            delta = np.linalg.norm(self.get_value_function(z_new) - self.get_value_function(z), ord=np.inf)
            
            if iterations % 100 == 0:
                print(f"Iter: {iterations}. Delta: {delta}")

            if delta < epsilon:
                break

            z = z_new
            iterations += 1
            deltas.append(delta)
        
        elapsed_time = time.time() - start_time
        
        self.z = z
        print(f"Converged in {iterations} iterations")
        return z, ModelBasedAlgsStats(elapsed_time, iterations, deltas, self.num_states, Vs, "PI")

    def get_value_function(self, z: np.ndarray = None) -> np.ndarray:
        """
        Compute the value function from the z vector.

        Args:
        - z (np.ndarray, optional): Value function vector. If None, uses self.z.

        Returns:
        - (np.ndarray): Computed value function.
        """
        if z is None:
            z = self.z
        
        result = np.log(z) * self.lmbda
        result[result == -np.inf] = np.finfo(np.float64).min
        
        return result
    
    
    def compute_value_function(self):
        """
        Compute the value function and derive the optimal policy.
        """
        if not hasattr(self, "z"):
            print("Will compute power iteration")
        _, self.stats = self.power_iteration()
        
        self.V = self.get_value_function()
        
        self.policy = self.get_optimal_policy(self.z)
        
    
    def to_MDP(self):
        """
        Convert the LMDP to an equivalent MDP.

        Returns:
        - mdp (MDP): The converted Markov Decision Process.
        """
        z, _ = self.power_iteration()
        
        control = self.get_control(z)
        print(f"Computing the MDP embedding of this LMDP...")
        # The minimum number of actions that can be done to achieve the same behaviour in an MDP.
        num_actions = np.max(np.sum(control > 0, axis=1))
        
        mdp = models.MDP.MDP(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            allowed_actions=[i for i in range(num_actions)],
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
        
        # np.random.seed(123) # TODO: remove when developing has finished.
        
        # Define the transition probability matrix of the MDP.
        for s in tqdm(range(self.num_non_terminal_states), desc="Generating transition matrix P", total=self.num_non_terminal_states):
            non_zero_positions = np.where(control[s, :] != 0)[0]
            probs = control[s, non_zero_positions].flatten()
            for i, a in enumerate(range(num_actions), start=1):
                mdp.P[s, a, non_zero_positions] = probs
                probs = np.roll(probs, shift=1, axis=0)
        
        V_lmdp = self.get_value_function(z)
        
        mdp.compute_value_function()
        V_mdp = mdp.V
        
        print("EMBEDDING ERROR:", np.mean(np.square(V_lmdp - V_mdp)))
        
        return mdp