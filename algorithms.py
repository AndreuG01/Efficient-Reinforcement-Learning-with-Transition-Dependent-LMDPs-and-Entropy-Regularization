import numpy as np
from MDP import MDP
import matplotlib.pyplot as plt


class QLearning:
    # Sutton and Barton page 131
    def __init__(self, mdp: MDP, alpha: float, gamma: float, epsilon: float, info_every: int = 1000):
        self.mdp = mdp
        self.Q = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        # print(self.mdp.R)
        # print(self.mdp.R[self.mdp.num_non_terminal_states, :])
        self.Q[self.mdp.num_non_terminal_states, :] = self.mdp.R[self.mdp.num_non_terminal_states, :] # TODO: Is it correct to initialize the Q function to the reward? 
        # Initialize Q(s, a), for all s in S+, a in A(s), arbitrarily except that Q(terminal ,·) = 0
        
        self.alpha = alpha # learning rate
        self.gamma = gamma
        self.epsilon = epsilon # exploration factor
        
        self.curr_state = self.mdp.s0
        
        self.reward = 0
        self.episode_terminated = False
        
        self.info_every = info_every
        
    
    def take_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            # With probability epsilon, explore (take random action)
            return np.random.choice(self.mdp.num_actions)
        
        # With probability (1 - epsilon), exploit (take action with best q-value)
        return np.argmax(self.Q[state, :])

    
    def step(self):
        action = self.take_action(self.curr_state)
        next_state, self.reward, is_terminal = self.mdp.transition(action, self.curr_state)
        
        # Q(s_t, a_t) = Q(s_t, a_t) + alpha * (r_{t+1} + gamma * max_{a_t+1}Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))
        self.Q[self.curr_state, action] = self.Q[self.curr_state, action] + self.alpha * (self.reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[self.curr_state, action])
        
        self.curr_state = next_state
        
        if is_terminal:
            self.curr_state = self.mdp.s0 # Return to the initial state
            self.episode_terminated = True
    
    
    def train(self, num_steps: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
        epoch = 0
        cumulative_reward = 0
        rewards = []
        episode_start = 0
        
        while epoch < num_steps:
            self.step()
            cumulative_reward += self.reward
            epoch += 1
            
            if self.episode_terminated:
                # print(f"(epoch: {epoch}) Episode finished in {epoch - episode_start} steps")
                rewards[episode_start:epoch] = [cumulative_reward] * (epoch - episode_start + 1)
                episode_start = epoch
                cumulative_reward = 0
                self.episode_terminated = False
            
            
            if epoch % self.info_every == 0:
                print(f"Epoch [{epoch} / {num_steps}]. Cumulative reward last episode: {rewards[episode_start - 1]}")
            
        policy = np.argmax(self.Q, axis=1) # Take the actions that have the best q-value for each state

        if not self.episode_terminated:
            rewards[episode_start:epoch] = [cumulative_reward] * (epoch - episode_start)
        
        # print(rewards)

        
        
        return self.Q, policy, rewards

