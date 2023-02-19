import unittest
import sys
import os
import numpy as np
import math
from typing import Tuple

# import code directory
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from electronTransportCode.MCEstimator import FluenceEstimator
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.utils import tuple2d

class TestMCFluenceEstimator(unittest.TestCase):
    def test_updateEstimatorA(self) -> None:
        # TEST A: energy jumps over multiple bins
        TOL = 1e-14
        domain = SimulationDomain(0, 1, 0, 1, 3, 3)
        Ebins = 10; Emin = 0; Emax = 1
        estimator = FluenceEstimator(domain, Emin, Emax, Ebins, spacing='lin')
        
        # set up arguments
        index = 0
        pos1 = np.array((5, 3))
        pos2 = np.array((0, 1))
        posTuple = (pos1, pos2)
        vecTuple = (pos1, pos2)
        stepsize = np.linalg.norm(pos2-pos1)  # type:ignore
        energyTuple = (0.95, 0.45)
        
        # call routine
        estimator.updateEstimator(posTuple, vecTuple, energyTuple, index)
        
        # Check result
        trueFluence = np.array((0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1))*stepsize
        self.assertEqual(np.allclose(trueFluence, estimator.scoreMatrix[:, index], rtol=TOL), True)
        
        return None
    
    def test_updateEstimatorB(self) -> None:
        # TEST B: energy jumps over one bin
        TOL = 1e-14
        domain = SimulationDomain(0, 1, 0, 1, 3, 3)
        Ebins = 10; Emin = 0; Emax = 1
        estimator = FluenceEstimator(domain, Emin, Emax, Ebins, spacing='lin')
        
        # set up arguments
        index = 0
        pos1 = np.array((5, 3))
        pos2 = np.array((0, 1))
        posTuple = (pos1, pos2)
        vecTuple = (pos1, pos2)
        stepsize = np.linalg.norm(pos2-pos1)  # type:ignore
        energyTuple = (0.55, 0.45)
        
        # call routine
        estimator.updateEstimator(posTuple, vecTuple, energyTuple, index)
        
        # Check result
        trueFluence = np.array((0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0))*stepsize
        for x, y in zip(trueFluence, estimator.scoreMatrix[:, index]):
            print(x, y)
        self.assertEqual(np.allclose(trueFluence, estimator.scoreMatrix[:, index], rtol=TOL), True)
        
        return None

    def test_updateEstimatorC(self) -> None:
        # TEST B: particle energy remains in the same bin
        TOL = 1e-14
        domain = SimulationDomain(0, 1, 0, 1, 3, 3)
        Ebins = 10; Emin = 0; Emax = 1
        estimator = FluenceEstimator(domain, Emin, Emax, Ebins, spacing='lin')
        
        # set up arguments
        index = 0
        pos1 = np.array((5, 3))
        pos2 = np.array((0, 1))
        posTuple = (pos1, pos2)
        vecTuple = (pos1, pos2)
        stepsize = np.linalg.norm(pos2-pos1)  # type:ignore
        energyTuple = (0.55, 0.56)
        
        # call routine
        estimator.updateEstimator(posTuple, vecTuple, energyTuple, index)
        
        # Check result
        trueFluence = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0))*stepsize
        for x, y in zip(trueFluence, estimator.scoreMatrix[:, index]):
            print(x, y)
        self.assertEqual(np.allclose(trueFluence, estimator.scoreMatrix[:, index], rtol=TOL), True)
        
        return None

if __name__ == '__main__':
    unittest.main()