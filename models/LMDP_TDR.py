import numpy as np
from scipy.sparse import csr_matrix
from collections.abc import Callable
from domains.grid import CustomGrid
from sys import getsizeof
from joblib import Parallel, delayed, cpu_count
import time
from tqdm import tqdm
from .MDP import MDP

class LMDP_TDR:
    def __init__(self, num_states: int, num_terminal_states: int, lmbda: int = 1, s0: int = 0, sparse_optimization: bool = True) -> None:
        self.num_states = num_states
        self.num_terminal_states = num_terminal_states
        self.num_non_terminal_states = self.num_states - self.num_terminal_states
        self.s0 = s0
        self.lmbda = lmbda
        self.sparse_optimization = sparse_optimization
        
        self.P: np.ndarray | csr_matrix = np.zeros((self.num_non_terminal_states, self.num_states))
        self.R = np.zeros((self.num_non_terminal_states, self.num_states))
    
    
    def generate_P(self, move: Callable, grid: CustomGrid, actions: list[int], num_threads: int = 10, benchmark: bool = False) -> float:
        """
        Generates the transition probability matrix (P) for the LMDP, based on the dynamics of the environment.

        Args:
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
        next_state = np.random.choice(self.num_states, p=self.P[state])
        
        return (
            next_state,
            self.R[next_state],
            next_state >= self.num_non_terminal_states
        )
    
    
    def get_control(self, z: np.ndarray, o: np.ndarray | csr_matrix) ->np.ndarray:
        
        if self.sparse_optimization:
            control = self.P.multiply(o).multiply(z)
        else:
            control = self.P * o * z
        
        control = control / np.sum(control, axis=1).reshape(-1, 1)
        # print(f"Control elements: {control.size}. Non zero: {np.count_nonzero(control)}")
        assert all(np.isclose(np.sum(control, axis=1), 1))
        
        return control
        
        
    
    
    def get_optimal_policy(self, z: np.ndarray, o: np.ndarray | csr_matrix, multiple_states: bool = False) -> np.ndarray:
        probs = self.get_control(z, o)
        if self.sparse_optimization:
            probs = probs.toarray()
        print(probs)
        return probs
        
        
        
    
    def transition_action(self, state: int, next_state: list[int]) -> list[int]:
        # LMDPs do not have actions. However, to be able to plot the policies, or interact with the environment, we need to convert the transitions into certain actions
        # (as long as the problem is deterministic)
        raise NotImplementedError("Implement in the subclass")

    
    def power_iteration(self, epsilon=1e-10) -> np.ndarray:
        if self.sparse_optimization:
            self.o = np.exp(self.R.toarray() * self.lmbda) # TODO: I have problems when this is a sparse matrix
            G = csr_matrix(self.P.multiply(self.o))
            
        else:
            self.o = np.exp(self.R / self.lmbda)
            G = self.P * self.o
    
    
        z = np.ones(self.num_states)
        iterations = 0
        delta = 0
        
        while True:
            z_new = G @ z
            z_new = np.concatenate((z_new, np.ones((self.num_terminal_states))))
           
            delta = np.linalg.norm(self.get_value_function(z_new) - self.get_value_function(z), ord=np.inf)
            
            if iterations % 100 == 0:
                print(f"Iter: {iterations}. Delta: {delta}")
        
            if delta < epsilon:
                break
            
            z = z_new
            iterations += 1
        
        self.z = z
        print(f"Converged in {iterations} iterations")
        return z
        
    
    def get_value_function(self, z: np.ndarray = None) -> np.ndarray:
        if z is None:
            z = self.z
        
        result = np.log(z) * self.lmbda
        result[result == -np.inf] = np.finfo(np.float64).min
        
        return result
    
    
    def compute_value_function(self):
        if not hasattr(self, "z"):
            print("Will compute power iteration")
            self.power_iteration()
        
        self.V = self.get_value_function()
        
        self.policy = self.get_optimal_policy(self.z, self.o)
        # self.policy_multiple_states = self.get_optimal_policy(self.z, self.o, multiple_states=True)
        
    
    def to_MDP(self) -> MDP:
        raise NotImplementedError("Not implemented yet")