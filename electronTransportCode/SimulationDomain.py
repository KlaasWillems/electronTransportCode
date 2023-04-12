from typing import Tuple, Optional
import numpy as np
from electronTransportCode.ProjectUtils import tuple3d
from electronTransportCode.Material import Material
from numba.experimental import jitclass
import numba as nb

# TODO: rename to yz terminology


class SimulationDomain:
    """A simulation domain object represents a rectangular domain [x_min, xmax] \times [y_min, y_max].
    This domain is divided into xbins columns and ybins rows.
    As particles traverse the simulation domain they belong to a cell in the domain. Cells are numbered row-wise as follows:
    | 6 | 7 | 8 |
    | 3 | 4 | 5 |
    | 0 | 1 | 2 |
    The number associated to a grid cell is referred to as the index.
    No periodic boundary conditions.
    """
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, xbins: int, ybins: int, material: Optional[Material]) -> None:
        """Initialize rectangular simulation domain

        Args:
            xmin (float): Smallest coordinate in horizontal direction
            xmax (float): Largest coordinate in horizontal direction
            ymin (float): Smallest coordinate in vertical direction
            ymax (float): Largest coordinate in vertical direction
            xbins (int): Amount of columns in rectangular grid
            ybins (int): Amount of rows in rectangular grid
            material (Material): domain is made up of uniform material
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xbins = xbins
        self.ybins = ybins
        self.xrange: np.ndarray = np.linspace(self.xmin, self.xmax, self.xbins+1).astype(np.float64)
        self.yrange: np.ndarray = np.linspace(self.ymin, self.ymax, self.ybins+1)
        self.dA: float = (self.xrange[1] - self.xrange[0])*(self.yrange[1] - self.yrange[0])
        self.material = material

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SimulationDomain):
            return self.xmin == other.xmin and self.xmax == other.xmax and self.ymin == other.ymin and self.ymax == other.ymax and self.xbins == other.xbins and self.ybins == other.ybins and self.material == other.material
        else:
            raise NotImplementedError()

    def getMaterial(self, index: int) -> Material:
        """Return material at grid cell index

        Args:
            index (int): Grid cell index

        Returns:
            Material:
        """
        if self.material is None:
            raise ValueError('Material is None.')
        else:
            return self.material

    def getCoord(self, index: int) -> Tuple[int, int]:
        """Return coordinate of grid cell in rectangular grid

        Args:
            index (int): grid cell index

        Returns:
            Tuple[int, int]: row and column
        """
        col = index % self.xbins
        row = (index-col) // self.xbins
        return row, col

    def getIndexPath(self, pos: tuple3d, vec: tuple3d) -> int:
        """Ruturn index associated to the grid cell in which the next path of the particle lies. If a particle lies in the interior of a cell, return the index of the grid cell. If the particle lies on a grid cell boundary, depending on direction of travel, return the index.

        Args:
            pos (tuple3d): Position tuple
            vec (tuple3d): Direction of travel tuple

        Returns:
            int: index
        """
        _, x, y = pos
        _, vecx, vecy = vec

        xAr = self.xrange == x
        yAr = self.yrange == y
        col = np.argmax(self.xrange >= x) - 1
        row = np.argmax(self.yrange >= y) - 1

        if np.any(xAr):  # Vertical boundary
            if vecx > 0:
                col += 1
            elif vecx == 0.0:
                raise NotImplementedError('Particle is moving along a vertical boundary')

        if np.any(yAr):  # Horizontal boundary
            if vecy > 0:
                row += 1
            elif vecx == 0.0:
                raise NotImplementedError('Particle is moving along a horizontal boundary')

        return row*self.xbins + col  # type:ignore

    def getNeighbourIndex(self, index: int, edge: int) -> int:
        """Given the index of the current cell, return the index of the cell that shares the vertex 'edge' with the current cell.
        -1 is returned if there is no neighbour in that direction. (no periodic boundaries)

        Args:
            index (int): index of current cell
            edge (int): 0 = left border, 1 = bottom border, 2 = right border, 3 = top border

        Returns:
            int: index of adjacent cell
        """
        assert index >= 0
        assert edge == 0 or edge == 1 or edge == 2 or edge == 3

        row, col = self.getCoord(index)

        if edge == 0 and col != 0: # left border
            return row*self.xbins + col - 1
        elif edge == 1 and row != 0:  # bottom border
            return (row - 1)*self.xbins + col
        elif edge == 2 and col != self.xbins - 1:  # right border
            return row*self.xbins + col + 1
        elif edge == 3 and row != self.ybins - 1:  # top border
            return (row + 1)*self.xbins + col
        return -1

    def checkDomainEdge(self, index, edge) -> bool:
        """Return true if edge of grid cell 'index' is a domain edge

        Args:
            index (int): grid cell index
            edge (int): 0 = left border, 1 = bottom border, 2 = right border, 3 = top border

        Returns:
            bool:
        """
        row, col = self.getCoord(index)
        if col == 0 and edge == 0:
            return True
        if col == self.xbins-1 and edge == 2:
            return True
        if row == 0 and edge == 1:
            return True
        if row == self.ybins-1 and edge == 3:
            return True
        return False

    def getCellEdgeInformation(self, pos: tuple3d, vec: tuple3d, index: int) -> tuple[float, bool, tuple3d]:
        """Return distance to nearest grid cell crossing, boolean and the grid cell crossing location
        The distance is computed by finding the intersection of the particle with the horizontal lines at xmin and xmax, and vertical lines at ymin and ymax. The smallest positive distance is returned.

        The second return argument is True if the edge the particle is heading to is also a domain edge.
        The third argument is the location of the next grid cell crossing.

        Args:
            pos (tuple2d): Position tuple
            vec (tuple2d): Direction of travel tuple
            index (int): index of current cell

        Returns:
            tuple[float, bool]: distance and domain edge boolean
        """
        assert self.getIndexPath(pos, vec) ==  index

        _, x0, y0 = pos
        _, vx, vy = vec
        row, col = self.getCoord(index)
        new_pos = np.array((0.0, 0.0, 0.0), dtype=float)

        # cell boundaries
        xmincell = self.xrange[col]
        xmaxcell = self.xrange[col+1]
        ymincell = self.yrange[row]
        ymaxcell = self.yrange[row+1]

        # particle must be in cell
        assert xmincell <= x0
        assert xmaxcell >= x0
        assert ymincell <= y0
        assert ymaxcell >= y0

        # compute distance to horizontal cell boundaries
        xEdge: int
        x_new_pos: float
        if vx < 0.0:  # Particle moving to left boundary
            tx = (xmincell - x0)/vx
            xEdge = 0
            x_new_pos = xmincell
        elif vx > 0.0:  # Particle moving to right boundary
            tx = (xmaxcell - x0)/vx
            xEdge = 2
            x_new_pos = xmaxcell
        else:  # particle is moving in the vertical direction
            xEdge = -1
            tx = np.infty
            x_new_pos = x0

        yEdge: int
        y_new_pos: float
        if vy < 0.0:  # Particle moving to lower boundary
            ty = (ymincell - y0)/vy
            yEdge = 1
            y_new_pos = ymincell
        elif vy > 0.0:  # Particle moving to upper boundary
            ty = (ymaxcell - y0)/vy
            yEdge = 3
            y_new_pos = ymaxcell
        else:  # particle is moving in the horizontal direction
            yEdge = -1
            ty = np.infty
            y_new_pos = y0

        tmin = min(tx, ty)
        edge = xEdge if tmin == tx else yEdge

        if tmin == tx:
            new_pos[1] = x_new_pos
            new_pos[2] = y0 + tmin*vec[2]
        else:
            new_pos[1] = x0 + tmin*vec[1]
            new_pos[2] = y_new_pos

        new_pos[0] = pos[0] + tmin*vec[0]

        return tmin, self.checkDomainEdge(index, edge), new_pos
