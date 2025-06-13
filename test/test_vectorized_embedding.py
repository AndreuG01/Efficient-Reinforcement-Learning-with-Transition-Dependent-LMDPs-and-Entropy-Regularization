import unittest
from domains.grid_world import GridWorldMDP, GridWorldActions
from domains.minigrid_env import MinigridMDP, MinigridActions
from utils.maps import Map, Maps
import numpy as np
from utils.coloring import TerminalColor


class VectorizedEmbeddingTester(unittest.TestCase):
    """
    Unit tests for vectorized embedding in GridWorldMDP and MinigridMDP.
    This class tests the equivalence of vectorized and iterative methods for converting MDPs to LMDPs.
    """
    def test_vectorized_equivalence(self):
        sizes = np.arange(2, 25, 1)
        for size in sizes:
            for prob in np.arange(0.1, 1.1, 0.1):
                print(f"Testing size [{TerminalColor.colorize(str(size), 'purple', bold=True)}/{max(sizes)}] and probability {TerminalColor.colorize(str(round(prob, 1)), 'purple', bold=True)}".ljust(100), end="\n")
                mdp = GridWorldMDP(
                    map=Map(grid_size=size),
                    behavior="stochastic",
                    stochastic_prob=prob,
                    temperature=1,
                    verbose=False
                )
                
                lmdp_vectorized, _, _ = mdp.to_LMDP_TDR(find_best_lmbda=True, vectorized=True)
                lmdp_iterative, _, _ = mdp.to_LMDP_TDR(find_best_lmbda=True, vectorized=False)
                self.assertEqual(lmdp_vectorized, lmdp_iterative)



if __name__ == "__main__":
    unittest.main()