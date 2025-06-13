# From the root directory (/Repo) run this file as python -m test.test_gridworld

import unittest
from domains.grid_world import GridWorldMDP, GridWorldActions, GridWorldLMDP, GridWorldLMDP_TDR
from utils.maps import Map, Maps
import numpy as np


class GridWorldMDPTester(unittest.TestCase):
    """
    Unit tests for the GridWorldMDP class.
    This class tests the initialization, value function computation, and policy visualization of the GridWorldMDP.
    """
    def setUp(self):
        print('\n', unittest.TestCase.id(self))
    
    def test_one_thread(self):
        """
        Tests the initialization of the GridWorldMDP with a single thread.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="deterministic",
            threads=1,
            verbose=False
        )
        self.assertIsNotNone(mdp)
    
    def test_multiple_thread(self):
        """
        Tests the initialization of the GridWorldMDP with 4 threads.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="deterministic",
            threads=4,
            verbose=False
        )
        self.assertIsNotNone(mdp)
    
    
    def test_value_function_det(self):
        """
        Tests the value function computation for a deterministic GridWorldMDP.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="deterministic",
            threads=4,
            verbose=False
        )
        mdp.compute_value_function(temp=mdp.temperature)
        np.testing.assert_array_equal(mdp.V, np.array([-30., -25., -20., -15., -25., -20., -15., -10., -20., -15., -10., -5., -15., -10., -5., 0.]))
    
    
    def test_value_function_stochastic(self):
        """
        Tests the value function computation for a stochastic GridWorldMDP.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="stochastic",
            threads=4,
            verbose=False
        )
        mdp.compute_value_function(temp=mdp.temperature)
        np.testing.assert_array_almost_equal(mdp.V, np.array([-34.0022293,  -28.64508644, -23.09023251, -17.71968537, -28.64508644, -23.26899982, -17.52785077, -11.96522066, -23.09023251, -17.52785077, -11.75966927,  -5.99051345, -17.71968537, -11.96522066,  -5.99051345, 0.]))
        

    def test_visualize_stochastic_mdp_policy_128(self):
        """
        Tests the visualization of the policy for a stochastic GridWorldMDP with a datatype of np.float128.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="stochastic",
            threads=4,
            verbose=False
        )
        
        try:
            mdp.visualize_policy(num_times=1, show_window=False)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")

    
    def test_visualize_deterministic_mdp_policy_128(self):
        """
        Tests the visualization of the policy for a deterministic GridWorldMDP with a datatype of np.float128.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="deterministic",
            threads=4,
            verbose=False
        )
        
        try:
            mdp.visualize_policy(num_times=3, show_window=False)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")
    
    
    def test_visualize_stochastic_mdp_policy_64(self):
        """
        Tests the visualization of the policy for a stochastic GridWorldMDP with a datatype of np.float64.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=10),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="stochastic",
            threads=4,
            verbose=False,
            dtype=np.float64
        )
        
        try:
            mdp.visualize_policy(num_times=1, show_window=False)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")

    
    def test_visualize_deterministic_mdp_policy_64(self):
        """
        Tests the visualization of the policy for a deterministic GridWorldMDP with a datatype of np.float64.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="deterministic",
            threads=4,
            verbose=False,
            dtype=np.float64
        )
        
        try:
            mdp.visualize_policy(num_times=3, show_window=True)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")


    def test_LMDP_policy(self):
        """
        Tests the conversion of a GridWorldMDP to an LMDP policy.
        """
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behavior="stochastic",
            verbose=False
        )
        
        mdp.P = np.array([
            [
                [0.3, 0.2, 0.1, 0.4],
                [0.4, 0.1, 0.1, 0.4],
                [0.7, 0.1, 0, 0.1],
                [0.7, 0.1, 0, 0.1]
            ],
            [
                [0.8, 0.1, 0, 0.1],
                [0.5, 0.3, 0.2, 0],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ],
            [
                [0.4, 0.2, 0.4, 0],
                [0, 0, 0.3, 0.7],
                [1, 0, 0, 0],
                [1, 0, 0, 0]
            ]
        ])
        mdp.policy = np.array([
            [0.4, 0.1, 0.2, 0.3],
            [0.5, 0.4, 0.1, 0],
            [0.1, 0.5, 0.2, 0.2]
        ])
        
        reference_policy = np.array([
            [0.51, 0.14, 0.05, 0.25],
            [0.61, 0.19, 0.11, 0.09],
            [0.44, 0.02, 0.19, 0.35]
        ])
        
        np.testing.assert_array_almost_equal(reference_policy, mdp.to_LMDP_policy())
    
    
    def test_reg_deterministic_lmdp_embedding(self):
        """
        Tests the embedding of an entropy-regularized deterministic GridWorldMDP into an LMDP.
        """
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behavior="deterministic",
            temperature=5,
            verbose=False
        )
        embedded_lmdp, _ = mdp.to_LMDP()
        self.assertIsNotNone(embedded_lmdp)
    
    
    def test_reg_stochastic_lmdp_embedding(self):
        """
        Tests the embedding of an entropy-regularized stochastic GridWorldMDP into an LMDP.
        """
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behavior="stochastic",
            temperature=3.5,
            stochastic_prob=0.1,
            verbose=False
        )
        embedded_lmdp, _ = mdp.to_LMDP()
        self.assertIsNotNone(embedded_lmdp)
    
    
    def test_reg_deterministic_lmdp_tdr_embedding(self):
        """
        Tests the embedding of an entropy-regularized deterministic GridWorldMDP into an LMDP with transition-dependent rewards.
        """
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behavior="deterministic",
            temperature=5,
            verbose=False
        )
        embedded_lmdp, _, _ = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
        self.assertIsNotNone(embedded_lmdp)
    
    
    def test_reg_stochastic_lmdp_tdr_embedding(self):
        """
        Tests the embedding of an entropy-regularized stochastic GridWorldMDP into an LMDP with transition-dependent rewards.
        """
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behavior="stochastic",
            temperature=3.5,
            stochastic_prob=0.1,
            verbose=False
        )
        embedded_lmdp, _, _ = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
        self.assertIsNotNone(embedded_lmdp)
    
    
class GridWorldLMDPTester(unittest.TestCase):
    pass

class GridWorldLMDP_TDRTester(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()