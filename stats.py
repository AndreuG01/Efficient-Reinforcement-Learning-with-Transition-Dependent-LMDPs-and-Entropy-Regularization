import matplotlib.pyplot as plt
import numpy as np

class ValueIterationStats:
    def __init__(self, time: float, rewards: list[float], iterations: int, deltas: list[float], num_states: int):
        self.time = time
        self.rewards = rewards
        self.iterations = iterations
        self.deltas = deltas
        self.num_states = num_states

    
    def print_statistics(self):
        print(f"Converged in {self.iterations} iterations.")
        print(f"Time taken: {self.time:.4f} seconds.")
    
    
    def plot_cum_reward(self, deterministic: bool, mdp: bool):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.arange(self.iterations), self.rewards, color="blue", marker=".")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.title(f"Cumulative reward for {'deterministic' if deterministic else 'stochastic'} {'MDP' if mdp else 'LMDP'}", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.grid()
        return fig
        