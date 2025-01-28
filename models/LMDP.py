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

    
    def power_iteration(self, iterations=100):
        G = np.diag(np.exp(self.R[:self.num_non_terminal_states]) / self.lmbda)
        z = np.ones((self.num_states, 1))
        
        print(f"GPz: {G.shape}, {self.P.shape}, {z.shape}")
        
        for i in range(iterations):
            z = G @ self.P @ z
            z = np.concatenate((z, np.ones((1, self.num_terminal_states))))
            print(f"Iteration {i}:\n{z}")
        
        print(f"V\n{np.log(z) * self.lmbda}")
        return z