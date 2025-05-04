# From the root directory (/Repo) run this file as python -m test.test_minigrid

import unittest
from domains.minigrid_env import MinigridActions, MinigridLMDP, MinigridLMDP_TDR, MinigridMDP
from utils.maps import Map
import numpy as np


class MiniGridMDPTester(unittest.TestCase):
    pass


class MiniGridLMDPTester(unittest.TestCase):
    #TODO: complete
    pass

class MiniGridLMDP_TDRTester(unittest.TestCase):
    #TODO: complete
    pass



if __name__ == "__main__":
    unittest.main()