import unittest
import sys
import os
import numpy as np
import math

# import code directory
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

from electronTransportCode.SimulationDomain import SimulationDomain

class TestSimulationDomain(unittest.TestCase):
    def test_returnIndex(self) -> None:
        domain = SimulationDomain(0, 1, 0, 1, 3, 3)
        
        y = 0.0
        for yi in range(3):
            x = 0
            y += 0.25
            for xi in range(3):
                x += 0.25
                index = domain.returnIndex(np.array((x, y)))
                self.assertEqual(index, 3*yi+xi)
        return None
    
    def test_returnNeighbourIndex(self) -> None:
        xbins = 4
        ybins = 3
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins)
        
        for xi in range(xbins):
            for yi in range(ybins):
                index = yi*xbins + xi
                for edge in range(4):
                    n1 = domain.returnNeighbourIndex(index, edge)
                    if edge == 0:  # Left edge
                        if xi == 0:
                            self.assertEqual(n1, -1)
                        else:
                            self.assertEqual(n1, yi*xbins + xi - 1)
                    elif edge == 1:  # bottom edge
                        if yi == 0:
                            self.assertEqual(n1, -1)
                        else:
                            self.assertEqual(n1, (yi - 1)*xbins + xi)
                    elif edge == 2:  # right edge
                        if xi == xbins - 1:
                            self.assertEqual(n1, -1)
                        else:
                            self.assertEqual(n1, yi*xbins + xi + 1)
                    elif edge == 3:  # top edge
                        if yi == ybins - 1:
                            self.assertEqual(n1, -1)
                        else:
                            self.assertEqual(n1, (yi + 1)*xbins + xi)
        
        return None
    
    # See getCellEdgeInformationTest.ggb for visualization.
    def test_getCellEdgeInformation1(self) -> None:
        TOL = 1e-14
        xbins = 4
        ybins = 3
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins)
        
        # Particle is in cell 6 and moves to the right. Closest cell is to the right with index 7.
        pos = np.array((0.7, 0.55))
        alfa = 0.0
        vec = np.array((np.cos(alfa), np.sin(alfa)))
        index = 6
        
        t, cell = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.05, rel_tol=TOL), True)
        self.assertEqual(cell, 7)
        
        # Particle is in cell 7 and moves to the right. The domain boundary is to the right.
        pos = np.array((0.95, 0.55))
        alfa = 0.0
        vec = np.array((np.cos(alfa), np.sin(alfa)))
        index = 7
        
        t, cell = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.05, rel_tol=TOL), True)
        self.assertEqual(cell, -1)

        return None
    
    def test_getCellEdgeInformation2(self) -> None:
        TOL = 1e-10
        xbins = 4
        ybins = 3
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins)
        pos = np.array((0.7, 0.55))
        index = 6
        
        # Particle is in cell 6 and is moving to the top left. Cell to which the particle is moving has index 10.
        alfa = np.radians(112)
        vec = np.array((np.cos(alfa), np.sin(alfa)))
        t, cell = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.1258290533124, rel_tol=TOL), True)
        self.assertEqual(cell, 10)
        
        # Particle is in cell 6 and is moving to the left. Cell to which the particle is moving has index 5.
        alfa = np.radians(202)
        vec = np.array((np.cos(alfa), np.sin(alfa)))
        t, cell = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.2157069485355, rel_tol=TOL), True)
        self.assertEqual(cell, 5)

        return None



if __name__ == '__main__':
    unittest.main()
