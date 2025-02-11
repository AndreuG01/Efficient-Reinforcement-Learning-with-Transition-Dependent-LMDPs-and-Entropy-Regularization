import matplotlib.pyplot as plt
import numpy as np

class ValueIterationStats:
    def __init__(self, time: float, iterations: int, deltas: list[float], num_states: int):
        self.time = time
        self.iterations = iterations
        self.deltas = deltas
        self.num_states = num_states

    
    def print_statistics(self):
        print(f"Converged in {self.iterations} iterations.")
        print(f"Time taken: {self.time:.4f} seconds.")
    
