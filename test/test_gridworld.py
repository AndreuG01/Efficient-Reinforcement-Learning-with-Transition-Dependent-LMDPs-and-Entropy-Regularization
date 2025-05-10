# From the root directory (/Repo) run this file as python -m test.test_gridworld

import unittest
from domains.grid_world import GridWorldMDP, GridWorldActions, GridWorldLMDP, GridWorldLMDP_TDR
from utils.maps import Map, Maps
import numpy as np


class GridWorldMDPTester(unittest.TestCase):
    
    def setUp(self):
        print('\n', unittest.TestCase.id(self))
    
    def test_one_thread(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="deterministic",
            threads=1,
            verbose=False
        )
        self.assertIsNotNone(mdp)
    
    def test_multiple_thread(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="deterministic",
            threads=4,
            verbose=False
        )
        self.assertIsNotNone(mdp)
    
    
    def test_value_function_det(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="deterministic",
            threads=4,
            verbose=False
        )
        mdp.compute_value_function(temp=mdp.temperature)
        np.testing.assert_array_equal(mdp.V, np.array([-30., -25., -20., -15., -25., -20., -15., -10., -20., -15., -10., -5., -15., -10., -5., 0.]))
    
    
    def test_value_function_stochastic(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="stochastic",
            threads=4,
            verbose=False
        )
        mdp.compute_value_function(temp=mdp.temperature)
        np.testing.assert_array_almost_equal(mdp.V, np.array([-34.0022293,  -28.64508644, -23.09023251, -17.71968537, -28.64508644, -23.26899982, -17.52785077, -11.96522066, -23.09023251, -17.52785077, -11.75966927,  -5.99051345, -17.71968537, -11.96522066,  -5.99051345, 0.]))
        

    def test_visualize_stochastic_mdp_policy_128(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="stochastic",
            threads=4,
            verbose=False
        )
        
        try:
            mdp.visualize_policy(num_times=1, show_window=False)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")

    
    def test_visualize_deterministic_mdp_policy_128(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="deterministic",
            threads=4,
            verbose=False
        )
        
        try:
            mdp.visualize_policy(num_times=3, show_window=False)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")
    
    
    def test_visualize_stochastic_mdp_policy_64(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=10),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="stochastic",
            threads=4,
            verbose=False,
            dtype=np.float64
        )
        
        try:
            mdp.visualize_policy(num_times=1, show_window=False)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")

    
    def test_visualize_deterministic_mdp_policy_64(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="deterministic",
            threads=4,
            verbose=False,
            dtype=np.float64
        )
        
        try:
            mdp.visualize_policy(num_times=3, show_window=True)
        except Exception as e:
            self.fail(f"visualize_policy() raised an exception {e}")


    def test_LMDP_policy(self):
        mdp = GridWorldMDP(
            map=Map(grid_size=4),
            allowed_actions=GridWorldActions.get_actions()[:4],
            behaviour="stochastic",
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
    
    
    
    # TODO: this does not currently work as lambda = 0 cannot be used in power iteration.
    # def test_deterministic_lmdp_embedding(self):
    #     mdp = GridWorldMDP(
    #         map=Maps.CLIFF,
    #         behaviour="deterministic",
    #         temperature=0,
    #         verbose=False
    #     )
    #     embedded_lmdp = mdp.to_LMDP()
    #     self.assertIsNotNone(embedded_lmdp)
    
    
    # TODO: this does not currently work as lambda = 0 cannot be used in power iteration.
    # def test_stochastic_lmdp_embedding(self):
    #     mdp = GridWorldMDP(
    #         map=Maps.CLIFF,
    #         behaviour="stochastic",
    #         temperature=0,
    #         stochastic_prob=0.1,
    #         verbose=False
    #     )
    #     embedded_lmdp = mdp.to_LMDP()
    #     self.assertIsNotNone(embedded_lmdp)
    
    
    def test_reg_deterministic_lmdp_embedding(self):
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behaviour="deterministic",
            temperature=5,
            verbose=False
        )
        embedded_lmdp = mdp.to_LMDP()
        self.assertIsNotNone(embedded_lmdp)
    
    
    def test_reg_stochastic_lmdp_embedding(self):
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behaviour="stochastic",
            temperature=3.5,
            stochastic_prob=0.1,
            verbose=False
        )
        embedded_lmdp = mdp.to_LMDP()
        self.assertIsNotNone(embedded_lmdp)
    
    
    # TODO: this does not currently work as lambda = 0 cannot be used in power iteration.
    # def test_deterministic_lmdp_tdr_embedding(self):
    #     mdp = GridWorldMDP(
    #         map=Maps.CLIFF,
    #         behaviour="deterministic",
    #         temperature=0,
    #         verbose=False,
    #         dtype=np.float64
    #     )
    #     embedded_lmdp = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
    #     self.assertIsNotNone(embedded_lmdp)
    
    
    # TODO: this does not currently work as lambda = 0 cannot be used in power iteration.
    # def test_stochastic_lmdp_tdr_embedding(self):
    #     mdp = GridWorldMDP(
    #         map=Maps.CLIFF,
    #         behaviour="stochastic",
    #         temperature=0,
    #         stochastic_prob=0.1,
    #         verbose=False,
    #         dtype=np.float64
    #     )
    #     embedded_lmdp = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
    #     self.assertIsNotNone(embedded_lmdp)
    
    
    def test_reg_deterministic_lmdp_tdr_embedding(self):
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behaviour="deterministic",
            temperature=5,
            verbose=False
        )
        embedded_lmdp, _ = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
        self.assertIsNotNone(embedded_lmdp)
    
    
    def test_reg_stochastic_lmdp_tdr_embedding(self):
        mdp = GridWorldMDP(
            map=Maps.CLIFF,
            behaviour="stochastic",
            temperature=3.5,
            stochastic_prob=0.1,
            verbose=False
        )
        embedded_lmdp, _ = mdp.to_LMDP_TDR(lmbda=mdp.temperature)
        self.assertIsNotNone(embedded_lmdp)
    
    
class GridWorldLMDPTester(unittest.TestCase):
    #TODO: complete
    pass

class GridWorldLMDP_TDRTester(unittest.TestCase):
    #TODO: complete
    pass


if __name__ == "__main__":
    unittest.main()