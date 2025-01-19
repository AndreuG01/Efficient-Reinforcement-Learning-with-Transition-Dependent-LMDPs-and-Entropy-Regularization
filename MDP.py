import numpy as np



class MDP:
    def __init__(self, num_states: int, num_terminal_states: int, num_actions: int, gamma: int = 1, s0: int = 0) -> None:
        """
        An MDP is a 5 tuple (S, A, T, R, gamma) where:
        - S is the state space (s_0 is the initial state)
        - A is the action space
        - P is the transition function (notation from Jonsson, Gómez 2016)
        - R is the reward function
        - gamma is the discount factor
        """
        assert 0 <= s0 <= num_states - 1, "Initial state must be a valid state"
        assert num_terminal_states < num_states, "There must be less terminal states than the overall number of states"
        assert 0 <= gamma <= 1, "Discount must be in the range [0, 1]"
        
        self.num_states = num_states
        self.num_terminal_states = num_terminal_states
        self.num_non_terminal_states = self.num_states - self.num_terminal_states
        self.num_actions = num_actions
        self.s0 = s0
        self.gamma = gamma
        
        # In the MDP superclass, P and R are set to 0. It will be in the subclasses where they will be set to the appropriate values for each domain.
        self.P = np.zeros((self.num_non_terminal_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))
        
    
    def transition(self, action: int, state: int) -> tuple[int, float, bool]:
        """
        action - action that needs to be taken
        state - current state where the agent is
        
        Returns the next state, the reward and whether the transitioned state is terminal or not
        """
        # With probability P, choose one of the next states
        next_state = np.random.choice(self.num_states, p=self.P[state, action])
        return (
            next_state,
            self.R[state, action],
            next_state >= self.num_non_terminal_states
        )
        
    def value_iteration(self, epsilon=1e-5):
        V = np.zeros(self.num_states)
        
        while True:
            delta = 0
            for s in range(self.num_non_terminal_states):
                v = V[s]
                V[s] = max(sum(self.P[s, a, s_next] * 
                           (self.R[s, a] + self.gamma * V[s_next])
                           for s_next in range(self.num_states)) for a in range(self.num_actions))
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break

        return V
    
    def get_optimal_policy(self, V):
        """
        Derive the optimal policy based on the computed value function V.
        Returns a list where each element corresponds to the optimal action for a state.
        """
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_non_terminal_states):  # Skip terminal states
            # Find the action that maximizes the expected utility
            policy[s] = np.argmax([
                sum(self.P[s, a, s_next] * (self.R[s, a] + self.gamma * V[s_next])
                    for s_next in range(self.num_states))
                for a in range(self.num_actions)
            ])
        return policy

    def print_rewards(self):
        """
        Print the rewards for each action in every state.
        """
        print("State | Action | Reward")
        print("-" * 25)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                print(f"{s:5d} | {a:6d} | {self.R[s, a]:.3f}")

    def print_action_values(self, V):
        """
        Print the value obtained at each state for each action.
        This uses the value function V to compute Q(s, a).
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

