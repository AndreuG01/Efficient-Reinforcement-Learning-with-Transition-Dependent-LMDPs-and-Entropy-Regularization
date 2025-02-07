import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable
from domains.grid import CellType, CustomGrid
from .MDP import MDP
from tqdm import tqdm

class LMDP(ABC):
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
    def __init__(self, num_states: int, num_terminal_states: int, lmbda: int = 1, s0: int = 1) -> None:
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
        
        self.P = np.zeros((self.num_non_terminal_states, self.num_states))
        self.R = np.zeros(self.num_states)
        
    
    
    def generate_P(self, pos: dict[int, list], move: Callable, grid: CustomGrid, actions: list[int]):
        """
        Generates the transition probability matrix (P) for the LMDP, based on the dynamics of the environment.

        Args:
        - pos (dict[int, list]): The different positions of the grid
        - move (Callable): A function that determines the next state based on the current state and action.
        The function signature should be `move(state: State, action: int) -> tuple[next_state: State, reward: float, done: bool]`.
        - grid (CustomGrid): The grid environment for which the transition matrix is being generated.
        - actions (list[int]): List of possible actions that can be taken in the environment (NOTE THAT THE LMDP DOES NOT HAVE ACTIONS INTO ACOUNT)
        """
        for state in tqdm(range(self.num_non_terminal_states), desc="Generating transition matrix P", total=self.num_non_terminal_states):
            for action in actions:
                if grid.is_cliff(grid.state_index_mapper[state]):
                    next_state = self.s0
                else:
                    next_state, _, terminal = move(pos[state], action)
                    # Convert from coordinate-like system (i, j) (grid format) to index based (idx) (matrix format)
                    if terminal:
                        next_state = grid.terminal_state_idx(next_state)
                    else:
                        next_state = pos.index(next_state)

                # TODO: add the equivalent to the deterministic in MDP, which would be a modifier of the transition probability
                self.P[state, next_state] += 1 / len(actions)
                
    
        assert all([np.isclose(np.sum(self.P[i, :]), 1) for i in range(self.P.shape[0])]), "Transition probabilities are not properly defined. They do not add up to 1 in every row"
        
        
        print(f"Generated matrix P with {self.P.size:,} elements")
    
    
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
        control = self.P * z
        control = control / np.sum(control, axis=1).reshape(-1, 1)
        
        assert all(np.isclose(np.sum(control, axis=1), 1))
        
        return control
        
    
    def get_optimal_policy(self, z: np.ndarray, multiple_states: bool = False) -> np.ndarray:
        """
        Compute the optimal policy based on the control matrix.

        Args:
        - z (np.ndarray): The transfomed value function vector.
        - multiple_states (bool): Whether to allow multiple optimal states (default is False).

        Returns:
        - policy (np.ndarray): The optimal policy.
        """
        policy = np.zeros(self.num_states, dtype=object)
        probs = self.get_control(z)
        
        if multiple_states:
            for i in range(probs.shape[0]):        
                policy[i] = [j for j in range(len(probs[i, :])) if probs[i, j] == np.max(probs[i, :])]  
        else:
            policy = probs.argmax(axis=1)
        
        return policy
        
    
    def transition_action(self, state: int, next_state: list[int]) -> list[int]:
        # LMDPs do not have actions. However, to be able to plot the policies, or interact with the environment, we need to convert the transitions into certain actions
        # (as long as the problem is deterministic)
        raise NotImplementedError("Implement in the subclass")
    
    
    def power_iteration(self, epsilon=0.62) -> np.ndarray:
        """
        Perform power iteration to compute the value function approximation.

        Args:
        - epsilon (float): Convergence threshold (default is 1e-20).

        Returns:
        - z (np.ndarray): Converged transformed value function vector.
        """
        G = np.diag(np.exp(self.R[:self.num_non_terminal_states]) / self.lmbda)
        z = np.ones(self.num_states)
        
        iterations = 0
        
        while True:
            delta = 0
            z_new = G @ self.P @ z
            # print(z_new)
            z_new = np.concatenate((z_new, np.ones((self.num_terminal_states))))
            
            delta = np.linalg.norm(self.get_value_function(z_new) - self.get_value_function(z))
    
            z = z_new
            
            if iterations % 10 == 0:
                print(f"Iter: {iterations}. Delta: {delta}")
            
            if delta < epsilon:
                break
            iterations += 1
        
        self.z = z
        
        return z

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
        return np.log(z) * self.lmbda
    
    
    def compute_value_function(self):
        """
        Compute the value function and derive the optimal policy.
        """
        if not hasattr(self, "z"):
            print("Will compute power iteration")
            self.power_iteration()
        
        self.V = self.get_value_function()
        
        self.policy = self.get_optimal_policy(self.z)
        self.policy_multiple_states = self.get_optimal_policy(self.z, multiple_states=True)
        
    
    def to_MDP(self) -> MDP:
        """
        Convert the LMDP to an equivalent MDP.

        Returns:
        - mdp (MDP): The converted Markov Decision Process.
        """
        epsilon = 1e-10 # To avoid division by 0 when dividing the control by P.
        control = self.get_control(self.power_iteration())
        
        # The minimum number of actions that can be done to achieve the same behaviour in an MDP.
        num_actions = np.max(np.sum(control > 0, axis=1))
        
        mdp = MDP(
            num_states=self.num_states,
            num_terminal_states=self.num_terminal_states,
            allowed_actions=[i for i in range(num_actions)],
        )
        
        # Define the reward function of the MDP.
        kl_term =  np.sum(control * np.log(control / (self.P + epsilon) + epsilon), axis=1) # TODO: maybe improve to avoid adding epsilon, and only modifying what is really necessary
        
        reward_non_terminal = self.R[:self.num_non_terminal_states] - self.lmbda * kl_term
        reward_non_terminal = np.tile(reward_non_terminal.reshape(-1, 1), (1, num_actions))
        
        mdp.R[:self.num_non_terminal_states, :] = reward_non_terminal
        
        # np.random.seed(123) # TODO: remove when developing has finished.
        
        # Define the transition probability matrix of the MDP.
        for s in range(self.num_non_terminal_states):
            non_zero_positions = np.where(control[s, :] != 0)[0]
            probs = control[s, non_zero_positions].flatten()
            for i, a in enumerate(range(num_actions), start=1):
                mdp.P[s, a, non_zero_positions] = probs
                # probs = np.roll(probs, shift=1, axis=0)
                probs = np.random.permutation(probs)
        
        
        V_lmdp = self.get_value_function(self.power_iteration())
        mdp.compute_value_function()
        V_mdp = mdp.V
        
        print("Error", np.mean(np.square(V_lmdp - V_mdp)))
        
        return mdp