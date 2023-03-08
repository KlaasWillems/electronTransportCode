# pylint: skip-file
import unittest
import sys
import os
import numpy as np
import math

sys.path.insert(0, os.path.abspath('..'))

from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import WaterMaterial

class TestSimulationDomain(unittest.TestCase):
    def test_returnIndexA(self) -> None:
        # test particle's in interior of domain
        domain = SimulationDomain(0, 1, 0, 1, 3, 3, WaterMaterial)

        y = 0.0
        for yi in range(3):
            x = 0.0
            y += 0.25
            for xi in range(3):
                x += 0.25
                index = domain.getIndexPath(np.array((1.0, x, y)), np.array((0.0, 1, 0)))
                self.assertEqual(index, 3*yi+xi)
        return None

    def test_checkIndex(self) -> None:
        xbins = 71
        ybins = 305
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins, WaterMaterial)

        domain.checkDomainEdge(60, 3)

        for xi in range(xbins):
            for yi in range(ybins):
                index = yi*xbins + xi
                row, col = domain.getCoord(index)
                self.assertEqual(xi, col)
                self.assertEqual(yi, row)

    def test_returnIndexB(self) -> None:
        # test particle's on a boundary
        domain = SimulationDomain(0, 1, 0, 1, 3, 3, WaterMaterial)

        x = domain.xrange[1]  # 1/3
        y = 0.1
        pos = np.array((0.0, x, y))

        vec1 = np.array((0.0, 1, 0))
        vec2 = np.array((0.0, -1, 0))

        self.assertEqual(domain.getIndexPath(pos, vec1), 1)
        self.assertEqual(domain.getIndexPath(pos, vec2), 0)

    def test_checkDomainEdge(self) -> None:
        xbins = 10
        ybins = 7
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins, WaterMaterial)

        domain.checkDomainEdge(60, 3)

        for xi in range(xbins):
            for yi in range(ybins):
                for edge in range(4):
                    index = yi*xbins + xi
                    n1 = domain.checkDomainEdge(index, edge)
                    if xi == 0 and edge == 0:
                        self.assertEqual(n1, True)
                    elif xi == xbins-1 and edge == 2:
                        self.assertEqual(n1, True)
                    elif yi == 0 and edge == 1:
                        self.assertEqual(n1, True)
                    elif yi == ybins-1 and edge == 3:
                        self.assertEqual(n1, True)
                    else:
                        self.assertEqual(n1, False)

    def test_returnNeighbourIndex(self) -> None:
        xbins = 4
        ybins = 3
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins, WaterMaterial)

        for xi in range(xbins):
            for yi in range(ybins):
                index = yi*xbins + xi
                for edge in range(4):
                    n1 = domain.getNeighbourIndex(index, edge)
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
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins, WaterMaterial)

        # Particle is in cell 6 and moves to the right. Closest cell is to the right with index 7.
        pos = np.array((0.0, 0.7, 0.55))
        alfa = 0.0
        vec = np.array((0.0, np.cos(alfa), np.sin(alfa)))
        index = 6

        t, domainEdgeBool, _ = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.05, rel_tol=TOL), True)
        self.assertEqual(domainEdgeBool, False)

        # Particle is in cell 7 and moves to the right. The domain boundary is to the right.
        pos = np.array((0.0, 0.95, 0.55))
        alfa = 0.0
        vec = np.array((0.0, np.cos(alfa), np.sin(alfa)))
        index = 7

        t, domainEdgeBool, _ = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.05, rel_tol=TOL), True)
        self.assertEqual(domainEdgeBool, True)

        return None

    def test_getCellEdgeInformation2(self) -> None:
        TOL = 1e-10
        xbins = 4
        ybins = 3
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins, WaterMaterial)
        pos = np.array((0.0, 0.7, 0.55))
        index = 6

        # Particle is in cell 6 and is moving to the top left. Cell to which the particle is moving has index 10.
        alfa = np.radians(112)
        vec = np.array((0.0, np.cos(alfa), np.sin(alfa)))
        t, domainEdgeBool, _ = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.1258290533124, rel_tol=TOL), True)
        self.assertEqual(domainEdgeBool, False)

        # Particle is in cell 6 and is moving to the left. Cell to which the particle is moving has index 5.
        alfa = np.radians(202)
        vec = np.array((0.0, np.cos(alfa), np.sin(alfa)))
        t, domainEdgeBool, _ = domain.getCellEdgeInformation(pos, vec, index)
        self.assertEqual(math.isclose(t, 0.2157069485355, rel_tol=TOL), True)
        self.assertEqual(domainEdgeBool, False)

        return None

    def test_getCellArea(self) -> None:
        TOL = 1e-15
        xbins = 4
        ybins = 3
        domain = SimulationDomain(0, 1, 0, 1, xbins, ybins, WaterMaterial)
        area = (1/3)*(1/4)
        self.assertEqual(math.isclose(area, domain.dA, rel_tol=TOL), True)


if __name__ == '__main__':
    unittest.main()
