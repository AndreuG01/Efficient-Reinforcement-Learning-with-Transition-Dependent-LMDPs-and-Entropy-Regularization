import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable
from domains.grid import CellType, CustomGrid

class LMDP(ABC):
    def __init__(self, num_states: int, num_terminal_states: int, lmbda: int = 1, s0: int = 1) -> None:
        self.num_states = num_states
        self.num_terminal_states = num_terminal_states
        self.num_non_terminal_states = self.num_states - self.num_terminal_states
        self.s0 = s0
        self.lmbda = lmbda
        
        self.P = np.zeros((self.num_non_terminal_states, self.num_states))
        self.R = np.zeros(self.num_states)
        
    
    
    def generate_P(self, pos: dict[int, list], move: Callable, grid: CustomGrid, actions: list[int]):
        for state in range(self.num_non_terminal_states):
            for action in actions:
                if grid.state_index_mapper[state] in pos[CellType.CLIFF]:
                    next_state = self.s0
                else:
                    next_state, _, _ = move(pos[CellType.NORMAL][state], action)
                    # Convert from coordinate-like system (i, j) (grid format) to index based (idx) (matrix format)
                    if next_state in pos[CellType.GOAL]:
                        next_state = pos[CellType.GOAL].index(next_state) + len(pos[CellType.NORMAL])
                    else:
                        next_state = pos[CellType.NORMAL].index(next_state)

                # TODO: add the equivalent to the deterministic in MDP, which would be a modifier of the transition probability
                self.P[state, next_state] += 1 / len(actions)
                
    
        assert all([np.sum(self.P[i, :]) == 1 for i in range(self.P.shape[0])]), "Transition probabilities are not properly defined. They do not add up to 1 in every row"
    
    
    @abstractmethod
    def _generate_R(self):
        raise NotImplementedError("Implement in the subclass.")
    
    
    def transition(self, state: int) -> tuple[int, float, bool]:
        next_state = np.random.choice(self.num_states, p=self.P[state])
        
        return (
            next_state,
            self.R[next_state],
            next_state >= self.num_non_terminal_states
        )
    
    
    def get_optimal_policy(self, z: np.ndarray, multiple_states: bool = False) -> np.ndarray:
        
        policy = np.zeros(self.num_states, dtype=object)
        normalization_term = np.sum(self.P @ self.z)
        
        probs = self.P * z / normalization_term
        
        if multiple_states:
            for i in range(probs.shape[0]):        
                policy[i] = [j for j in range(len(probs[i, :])) if probs[i, j] == np.max(probs[i, :])]  
        else:
            policy = probs.argmax(axis=1)
        
        return policy
        
    
    @abstractmethod
    def policy_to_action(self, state: int, next_state: list[int]) -> list[int]:
        # LMDPs do not have actions. However, to be able to plot the policies, or interact with the environment, we need to convert the transitions into certain actions
        # (as long as the problem is deterministic)
        raise NotImplementedError("Implement in the subclass")
    
    
    def power_iteration(self, epsilon=1e-20):
        G = np.diag(np.exp(self.R[:self.num_non_terminal_states]) / self.lmbda)
        z = np.ones(self.num_states)
        epochs = 0
        while True:
            delta = 0
            z_new = G @ self.P @ z
            # print(z_new)
            z_new = np.concatenate((z_new, np.ones((self.num_terminal_states))))
            
            delta = np.linalg.norm(self.get_value_function(z_new) - self.get_value_function(z))
    
            z = z_new
            
            if delta < epsilon:
                break
            epochs += 1
            
        print(epochs)
        
        self.z = z
        
        return z

    def get_value_function(self, z: np.ndarray = None):
        if z is None:
            z = self.z
        return np.log(z) * self.lmbda
    
    
    def compute_value_function(self):
        if not hasattr(self, "z"):
            print("Will compute power iteration")
            self.power_iteration()
        
        self.V = self.get_value_function()
        
        self.policy = self.get_optimal_policy(self.z)
        self.policy_multiple_states = self.get_optimal_policy(self.z, multiple_states=True)
        
        