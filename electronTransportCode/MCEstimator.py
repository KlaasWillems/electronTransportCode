from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import numpy.typing as npt
from mpi4py import MPI
import numpy as np
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.ProjectUtils import ERE, tuple3d


class MCEstimator(ABC):
    """Monte Carlo estimator for particle simulations
    """
    def __init__(self, simDomain: SimulationDomain) -> None:
        self.simDomain: SimulationDomain = simDomain
        self.scoreMatrix: npt.NDArray

    @abstractmethod
    def updateEstimator(self, posTuple: tuple[tuple3d, tuple3d], vecTuple: tuple[tuple3d, tuple3d], energyTuple: tuple[float, float], index: int) -> None:
        """Score quantity of interest after a collision

        Args:
            posTuple (tuple[tuple3d, tuple3d]): old_position and new position of particle after collision
            vecTuple (tuple[tuple3d, tuple3d]): old direction of travel and new position of travel after collision
            energyTuple (tuple[float, float]): old energy of particle and new energy of particle after collision
            index (int): cell identifier in simulation domain
        """

    @abstractmethod
    def getEstimator(self) -> np.ndarray:
        """Return estimated quantity of interest

        Returns:
            np.ndarray: quantity of interest
        """

    @abstractmethod
    def __add__(self, other: MCEstimator) -> MCEstimator:
        """Add results of this estimator to results of another estimator.
        """

    @abstractmethod
    def combineEstimators(self, particles_per_proc: int, root: int = 0) -> None:
        """Combine estimators from multiple processors into one.
        """


class TrackEndEstimator(MCEstimator):
    """Log quatities of interest at the point when the particle dies.
    """
    def __init__(self, simDomain: SimulationDomain, nb_particles: int, setting: str) -> None:
        """
        Args:
            simDomain (SimulationDomain): Simulation Domain object
            nb_particles (int): Amount of particles that will be simulated.
            setting (str, optional): A string to indicate which quantity of interest to log at particle death. E.g. 'y' will log the particles y coorindate at particle death.
        """
        super().__init__(simDomain)
        self.nb_particles = nb_particles
        self.scoreMatrix = np.zeros((self.nb_particles, ))
        self.index: int = 0
        self.setting = setting

    def updateEstimator(self, posTuple: tuple[tuple3d, tuple3d], vecTuple: tuple[tuple3d, tuple3d], energyTuple: tuple[float, float], index: int) -> None:
        _, new_energy = energyTuple
        _, new_pos = posTuple
        if new_energy == 0.0:
            if self.setting == 'x':
                self.scoreMatrix[self.index] = new_pos[0]
            elif self.setting == 'y':
                self.scoreMatrix[self.index] = new_pos[1]
            elif self.setting == 'z':
                self.scoreMatrix[self.index] = new_pos[2]
            elif self.setting == 'r':
                self.scoreMatrix[self.index] = np.sqrt(new_pos[0]**2 + new_pos[1]**2 + new_pos[2]**2)
            elif self.setting == 'rz':
                self.scoreMatrix[self.index] = np.sqrt(new_pos[0]**2 + new_pos[1]**2) # Distance to z-axis
            elif self.setting == 'rx':
                self.scoreMatrix[self.index] = np.sqrt(new_pos[2]**2 + new_pos[1]**2)  # Distance to x-axis
            elif self.setting == 'ry':
                self.scoreMatrix[self.index] = np.sqrt(new_pos[0]**2 + new_pos[2]**2)  # Distance to y-axis
            elif self.setting == 'rz-Linesource':
                z = new_pos[2]
                if z > -0.3 and z < 0.3:
                    self.scoreMatrix[self.index] = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
                else:
                    self.index -= 1
            else:
                raise NotImplementedError('Requested estimator is not implemented. Check for typos.')
            self.index += 1

    def getEstimator(self) -> np.ndarray:
        return self.scoreMatrix

    def getDensityData(self, bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        binVal, binEdge = np.histogram(self.scoreMatrix, bins=bins, density=True)
        binCenter = (binEdge[:-1] + binEdge[1:])/2.0
        return binCenter, binVal

    def __add__(self, other: MCEstimator) -> MCEstimator:
        if isinstance(other, TrackEndEstimator):
            assert self.simDomain == other.simDomain
            assert self.setting == other.setting
            new = TrackEndEstimator(self.simDomain, other.nb_particles+self.nb_particles, self.setting)
            new.scoreMatrix = np.concatenate((self.scoreMatrix, other.scoreMatrix))
            new.index = self.nb_particles + other.nb_particles
            return new
        else:
            raise ValueError('Not adding the same estimators!')

    def combineEstimators(self, particles_per_proc: int, root: int = 0) -> None:
        myrank = MPI.COMM_WORLD.Get_rank()
        nproc = MPI.COMM_WORLD.Get_size()

        # gather estimator
        if myrank == root:
            recvbuf = np.empty((nproc, particles_per_proc), dtype=float)
        else:
            recvbuf = None  # type: ignore

        MPI.COMM_WORLD.Gather(self.scoreMatrix, recvbuf, root=0)

        if myrank == root:
            assert recvbuf is not None
            nb_particles = int(nproc*particles_per_proc)
            self.scoreMatrix = recvbuf.reshape((nb_particles, ))
            self.nb_particles = nb_particles
            self.index = nb_particles


class DoseEstimator(MCEstimator):
    """Score dose [MeV/g] at each collision and grid cell crossing
    """
    def __init__(self, simDomain: SimulationDomain) -> None:
        super().__init__(simDomain)
        self.scoreMatrix = np.zeros((self.simDomain.xbins*self.simDomain.ybins, ))  # Energy relative to ERE

    def updateEstimator(self, posTuple: tuple[tuple3d, tuple3d], vecTuple: tuple[tuple3d, tuple3d], energyTuple: tuple[float, float], index: int) -> None:
        """Score energy at cell

        Args:
            posTuple (tuple[tuple3d, tuple3d]): old_position and new position of particle after collision
            vecTuple (tuple[tuple3d, tuple3d]): old direction of travel and new position of travel after collision
            energyTuple (tuple[float, float]): old energy of particle and new energy of particle after collision
            index (int): cell identifier in simulation domain
        """
        energy, newEnergy = energyTuple
        self.scoreMatrix[index] += energy-newEnergy

    def getEstimator(self) -> np.ndarray:
        """Return dose [MeV/g] on whole domain.
        """
        out = np.copy(self.scoreMatrix)*ERE  # To MeV
        for index in range(self.simDomain.xbins*self.simDomain.ybins):
            out[index] /= self.simDomain.getMaterial(index).rho*self.simDomain.dA  # To Mev/g
        return out

    def __add__(self, other: MCEstimator) -> MCEstimator:
        if isinstance(other, DoseEstimator):
            assert self.simDomain == other.simDomain
            new = DoseEstimator(self.simDomain)
            new.scoreMatrix = self.scoreMatrix + other.scoreMatrix
            return new
        else:
            raise ValueError('Not adding the same estimators!')

    def combineEstimators(self, particles_per_proc: int, root: int = 0) -> None:
        myrank = MPI.COMM_WORLD.Get_rank()
        nproc = MPI.COMM_WORLD.Get_size()

        # gather estimator
        if myrank == root:
            recvbuf = np.empty((nproc, self.simDomain.xbins*self.simDomain.ybins), dtype=float)
        else:
            recvbuf = None  # type: ignore

        MPI.COMM_WORLD.Gather(self.scoreMatrix, recvbuf, root=0)

        if myrank == root:
            assert recvbuf is not None
            nb_particles = int(nproc*particles_per_proc)
            self.scoreMatrix = np.sum(recvbuf, axis=0)
            self.nb_particles = nb_particles
            self.index = nb_particles


class FluenceEstimator(MCEstimator):
    """Score particle density at each collision and grid cell crossing. Energy variable is discretized in bins.
    """
    def __init__(self, simDomain: SimulationDomain, Emin: float, Emax: float, Ebins: int, spacing: str = 'lin') -> None:
        """ Creates energy bins for fluence estimator. Energy bins include the left edge, except for the final bin.
                e.g. [0, 0.2), [0.2, 0.4), ... , [0.8, 1.0]
        Args:
            simDomain (SimulationDomain): SimulationDomain object
            Emin (float): Energy cutOff value
            Emax (float): Max particle energy from source
            Ebins (int): Amount of bins de energy range is divided into
            spacing (str, optional): Logarithmic or linear spacing of energy bins. Defaults to 'lin'.

        Raises:
            ValueError: If spacing is not "lin" or "log".
        """
        super().__init__(simDomain)
        self.Ebins = Ebins  # In electron rest energy
        self.Emin = Emin
        self.Emax = Emax
        self.spacing = spacing
        self.scoreMatrix = np.zeros((self.Ebins, self.simDomain.xbins*self.simDomain.ybins), dtype=float)  # cell index is column index, energy bin index is row index

        if spacing == 'lin':
            self.Erange = np.linspace(self.Emin, self.Emax, self.Ebins+1)
        elif spacing == 'log':
            self.Erange = np.logspace(np.log10(self.Emin), np.log10(self.Emax), self.Ebins+1)
        else:
            raise ValueError('Spacing argument is invalid. Should be "lin" or "log".')

    def __add__(self, other: MCEstimator) -> MCEstimator:
        if isinstance(other, FluenceEstimator):
            assert self.simDomain == other.simDomain
            assert self.Ebins == other.Ebins and self.Emin == other.Emin and self.Emax == other.Emax and self.spacing == other.spacing
            new = FluenceEstimator(self.simDomain, self.Emin, self.Emax, self.Ebins, self.spacing)
            new.scoreMatrix = self.scoreMatrix + other.scoreMatrix
            return new
        else:
            raise ValueError('Not adding the same estimators!')

    def getEstimator(self) -> np.ndarray:
        return self.scoreMatrix/self.simDomain.dA  # type: ignore

    def updateEstimator(self, posTuple: tuple[tuple3d, tuple3d], vecTuple: tuple[tuple3d, tuple3d], energyTuple: tuple[float, float], index: int) -> None:

        # Unpack
        energy, newEnergy = energyTuple
        pos, new_pos = posTuple

        assert self.Emin <= energy <= self.Emax
        assert self.Emin <= newEnergy <= self.Emax

        dE = energy - newEnergy
        stepsize: float = np.sqrt((pos[0] - new_pos[0])**2 + (pos[1] - new_pos[1])**2 + (pos[2] - new_pos[2])**2)

        # bin the energies
        bin1, bin2 = np.searchsorted(self.Erange, (energy, newEnergy), side='right')
        # Under normal circumstances, 'bin' ranges from 1 to self.Ebins. In case that energy == self.Emax, bin is self.Ebins + 1.
        # In case energy < self.Emin, bin is 0.

        # Make final bin include right edge
        if bin1 == self.Ebins+1:
            bin1 = self.Ebins
        if bin2 == self.Ebins+1:
            bin2 = self.Ebins

        # score
        if bin1 == bin2:  # same energy bin
            self.scoreMatrix[bin1-1, index] += stepsize
        elif bin1-bin2 == 1:  # one bin apart: energy > Emid > newEnergy
            Emid = self.Erange[bin2]
            self.scoreMatrix[bin1-1, index] += stepsize*(energy - Emid)/dE
            self.scoreMatrix[bin2-1, index] += stepsize*(Emid - newEnergy)/dE
        else:  # multiple bins apart
            Earray = np.zeros((bin1-bin2+2, ))
            Earray[1:-1] = self.Erange[bin2:bin1]
            Earray[0] = newEnergy
            Earray[-1] = energy
            diff = np.diff(Earray)
            for diffIndex, Ebin in enumerate(range(bin2, bin1+1)):
                self.scoreMatrix[Ebin-1, index] += stepsize*diff[diffIndex]/dE

    def combineEstimators(self, particles_per_proc: int, root: int = 0) -> None:
        raise NotImplementedError
