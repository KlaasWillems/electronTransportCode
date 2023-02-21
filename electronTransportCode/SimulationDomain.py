import numpy as np
from electronTransportCode.ProjectUtils import tuple2d
from electronTransportCode.Material import Material, WaterMaterial


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
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, xbins: int, ybins: int) -> None:
        """Initialize rectangular simulation domain

        Args:
            xmin (float): Smallest coordinate in horizontal direction
            xmax (float): Largest coordinate in horizontal direction
            ymin (float): Smallest coordinate in vertical direction
            ymax (float): Largest coordinate in vertical direction
            xbins (int): Amount of columns in rectangular grid
            ybins (int): Amount of rows in rectangular grid
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xbins = xbins
        self.ybins = ybins
        self.xrange: np.ndarray = np.linspace(self.xmin, self.xmax, self.xbins+1)
        self.yrange: np.ndarray = np.linspace(self.ymin, self.ymax, self.ybins+1)
        self.dA: float = (self.xrange[1] - self.xrange[0])*(self.yrange[1] - self.yrange[0])

    def getMaterial(self, index: int) -> Material:
        """Return material at grid cell index

        Args:
            index (int): Grid cell index

        Returns:
            Material:
        """
        # All water simulation domain. TODO: expand capabilities.
        return WaterMaterial

    def getIndexPath(self, pos: tuple2d, vec: tuple2d) -> int:
        """Ruturn index associated to the grid cell in which the next path of the particle lies. If a particle lies in the interior of a cell, return the index of the grid cell. If the particle lies on a grid cell boundary, depending on direction of travel, return the index.

        Args:
            pos (tuple2d): Position tuple

        Returns:
            int: index
        """
        x, y = pos
        vecx, vecy = vec

        xAr = self.xrange == x
        yAr = self.yrange == y
        col = np.argmax(self.xrange >= x) - 1
        row = np.argmax(self.yrange >= y) - 1

        if any(xAr):  # Vertical boundary
            if vecx > 0:
                col += 1
            elif vecx == 0.0:
                raise NotImplementedError('Particle is moving along a vertical boundary')

        if any(yAr):  # Horizontal boundary
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

        col = index % self.xbins
        row = (index-col) // self.ybins

        if edge == 0 and col != 0: # left border
            return row*self.xbins + col - 1
        elif edge == 1 and row != 0:  # bottom border
            return (row - 1)*self.xbins + col
        elif edge == 2 and col != self.xbins - 1:  # right border
            return row*self.xbins + col + 1
        elif edge == 3 and row != self.ybins - 1:  # top border
            return (row + 1)*self.xbins + col
        return -1


    def getCellEdgeInformation(self, pos: tuple2d, vec: tuple2d, index: int) -> tuple[float, int]:
        """Return distance to nearest grid cell crossing and an integer.
        The distance is computed by finding the intersection of the particle with the horizontal lines at xmin and xmax, and vertical lines at ymin and ymax. The smallest positive distance is returned.

        The extra integer is the index of the neighbouring cell if the particle were to be placed at the closest grid cell crossing. If the integer is negative, there is no neighbour in that direction (particle is at the boundary).

        Args:
            pos (tuple2d): Position tuple
            vec (tuple2d): Direction of travel tuple
            index (int): index of current cell

        Returns:
            tuple[float, int]: distance and domain edge boolean
        """
        assert self.getIndexPath(pos, vec) ==  index

        x0, y0 = pos
        vx, vy = vec
        col = index % self.xbins
        row = (index-col) // self.ybins

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
        if vx < 0.0:  # Particle moving to left boundary
            tx = (xmincell - x0)/vx
            xEdge = 0
        elif vx > 0.0:  # Particle moving to right boundary
            tx = (xmaxcell - x0)/vx
            xEdge = 2
        else:
            xEdge = -1
            tx = np.infty

        yEdge: int
        if vy < 0.0:  # Particle moving to lower boundary
            ty = (ymincell - y0)/vy
            yEdge = 1
        elif vy > 0.0:  # Particle moving to upper boundary
            ty = (ymaxcell - y0)/vy
            yEdge = 3
        else:
            yEdge = -1
            ty = np.infty

        tmin = min(tx, ty)
        edge = xEdge if tmin == tx else yEdge

        return tmin, self.getNeighbourIndex(index, edge)
