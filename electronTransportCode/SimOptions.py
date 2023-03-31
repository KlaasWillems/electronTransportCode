from abc import ABC, abstractmethod
import numpy as np
from electronTransportCode.ProjectUtils import tuple3d


class SimOptions(ABC):
    """Object which encapsulates initial condition and random number generator for Monte Carlo Simulation
    """
    def __init__(self, minEnergy: float, rngSeed: int = 12) -> None:
        self.minEnergy: float = minEnergy
        self.rng: np.random.Generator = np.random.default_rng(rngSeed)

        # Once self.minEnergy is reached, deposit the energy at position of 'death'
        self.DEPOSIT_REMAINDING_E_LOCALLY = True

    @abstractmethod
    def initialDirection(self) -> tuple3d:
        """Sample initial direction of particle
        """

    @abstractmethod
    def initialPosition(self) -> tuple3d:
        """Sample initial position of particle
        """

    @abstractmethod
    def initialEnergy(self) -> float:
        """Sample initial energy of particle
        """


class WaterPhantom(SimOptions):
    """Initial conditions for water phantom experiment
    """
    def __init__(self, minEnergy: float, eSource: float, xVariance: float, rngSeed: int = 12) -> None:
        super().__init__(minEnergy, rngSeed)
        self.eSource = eSource
        self.xVariance = xVariance

    def initialDirection(self) -> tuple3d:
        """Particle moving to the right
        """
        return np.array((0.0, 1.0, 1.0))*np.sqrt(2)/2

    def initialPosition(self) -> tuple3d:
        """Initial position at origin
        """
        xSample = self.rng.normal(scale=np.sqrt(self.xVariance))
        ySample = self.rng.normal(scale=np.sqrt(self.xVariance))
        return np.array((0.0, xSample, ySample))

    def initialEnergy(self) -> float:
        """Constant particle energy source at self.Esource
        """
        return self.eSource


class PointSource(SimOptions):
    """Initial conditions for point source benchmark
    """
    def __init__(self, minEnergy: float, rngSeed: int, eSource: float) -> None:
        super().__init__(minEnergy, rngSeed)
        self.eSource = eSource

    def initialDirection(self) -> tuple3d:
        """Uniformly distributed initial direction
        """
        # isotropic cos(theta)
        cost = self.rng.uniform(low=-1, high=1)
        sint = np.sqrt(1 - cost**2)  # scatter left or right with equal probability

        # uniformly distributed azimuthal scattering angle
        phi = self.rng.uniform(low=0.0, high=2*np.pi)
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        return np.array((sint*cosphi, sint*sinphi, cost), dtype=float)

    def initialPosition(self) -> tuple3d:
        """Initial position at origin
        """
        return np.array((0.0, 0.0, 0.0), dtype=float)

    def initialEnergy(self) -> float:
        """Constant particle energy source at self.Esource
        """
        return self.eSource

class KDTestSource(PointSource):
    def __init__(self, minEnergy: float, rngSeed: int, eSource: float) -> None:
        super().__init__(minEnergy, rngSeed, eSource)

    def initialDirection(self) -> tuple3d:
        return super().initialDirection()

    def initialPosition(self) -> tuple3d:
        # Gaussian at -loc and loc
        loc = 3
        s = self.rng.uniform(low=-0.5, high=0.5)
        x = self.rng.normal(loc=loc*np.sign(s))
        return np.array((x, 0.0, 0.0), dtype=float)


class DiffusionPointSource(PointSource):
    """Initial conditions for diffusion limit point source benchmark. particle's x-coordinate is random normally distributed with mean 'loc' and standard deviation 'scale'.
    """
    def __init__(self, minEnergy: float, rngSeed: int, eSource: float, loc: float, std: float) -> None:
        """
        Args:
            loc (float): Mean of normal distribution of particle's x-coordinate
            scale (float): Standard deviation of normal distribution of particle's x-coordinate
        """
        super().__init__(minEnergy, rngSeed, eSource)
        self.loc = loc
        self.std = std

    def initialPosition(self) -> tuple3d:
        """Initial position at origin
        """
        return np.array((self.rng.normal(loc=self.loc, scale=self.std), 0.0, 0.0), dtype=float)


class LineSource(PointSource):
    def __init__(self, minEnergy: float, rngSeed: int, eSource: float, xmin: float, xmax: float) -> None:
        super().__init__(minEnergy, rngSeed, eSource)
        self.xmin = xmin
        self.xmax = xmax

    def initialPosition(self) -> tuple3d:
        """Source along the z-axis
        """
        return np.array((0.0, 0.0, self.rng.uniform(low=self.xmin, high=self.xmax, size=1)), dtype=float)
