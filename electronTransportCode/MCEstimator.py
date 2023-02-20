from abc import ABC, abstractmethod
import numpy as np
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.ProjectUtils import ERE, tuple2d

# TODO: Estimators up to now assume grid cell crossing events. In KD Monte Carlo, particle can cross grid cell boundaries.


class MCEstimator(ABC):
    """Monte Carlo estimator for particle simulations
    """
    def __init__(self, simDomain: SimulationDomain) -> None:
        self.simDomain = simDomain

    @abstractmethod
    def updateEstimator(self, posTuple: tuple[tuple2d, tuple2d], vecTuple: tuple[tuple2d, tuple2d], energyTuple: tuple[float, float], index: int) -> None:
        """Score quantity of interest after a collision

        Args:
            posTuple (tuple[tuple2d, tuple2d]): old_position and new position of particle after collision
            vecTuple (tuple[tuple2d, tuple2d]): old direction of travel and new position of travel after collision
            energyTuple (tuple[float, float]): old energy of particle and new energy of particle after collision
            index (int): cell identifier in simulation domain
        """

    @abstractmethod
    def getEstimator(self) -> np.ndarray:
        """Return estimated quantity of interest

        Returns:
            np.ndarray: quantity of interest
        """


class DoseEstimator(MCEstimator):
    """Score dose [MeV/g] at each collision and grid cell crossing
    """
    def __init__(self, simDomain: SimulationDomain) -> None:
        super().__init__(simDomain)
        self.scoreMatrix = np.zeros((self.simDomain.xbins*self.simDomain.ybins, ))  # Energy relative to ERE

    def updateEstimator(self, posTuple: tuple[tuple2d, tuple2d], vecTuple: tuple[tuple2d, tuple2d], energyTuple: tuple[float, float], index: int) -> None:
        """Score energy at cell

        Args:
            posTuple (tuple[tuple2d, tuple2d]): old_position and new position of particle after collision
            vecTuple (tuple[tuple2d, tuple2d]): old direction of travel and new position of travel after collision
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


class FluenceEstimator(MCEstimator):
    """Score particle density at each collision and grid cell crossing. Energy variable is discretized in bins.
    """
    def __init__(self, simDomain: SimulationDomain, Emin: float, Emax: float, Ebins: int, spacing: str = 'lin') -> None:
        """
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

        self.scoreMatrix = np.zeros((self.Ebins, self.simDomain.xbins*self.simDomain.ybins))  # cell index is column index, energy bin index is row index

        if spacing == 'lin':
            self.Erange = np.linspace(self.Emin, self.Emax, self.Ebins+1)
        elif spacing == 'log':
            self.Erange = np.logspace(np.log10(self.Emin), np.log10(self.Emax), self.Ebins+1)
        else:
            raise ValueError('Spacing argument is invalid. Should be "lin" or "log".')

    def getEstimator(self) -> np.ndarray:
        return self.scoreMatrix/self.simDomain.dA

    def updateEstimator(self, posTuple: tuple[tuple2d, tuple2d], vecTuple: tuple[tuple2d, tuple2d], energyTuple: tuple[float, float], index: int) -> None:

        # Unpack
        energy, newEnergy = energyTuple
        pos, new_pos = posTuple

        dE = energy - newEnergy
        stepsize: float = np.linalg.norm(pos - new_pos)

        # bin the energies
        bin1, bin2 = np.digitize((energy, newEnergy), self.Erange)

        # score
        if bin1 == bin2:  # same energy bin
            self.scoreMatrix[bin1-1, index] += stepsize
        elif bin1-bin2 == 1:  # one bin apart: energy < Emind < newEnergy
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
