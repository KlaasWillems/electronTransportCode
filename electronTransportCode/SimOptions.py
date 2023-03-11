from abc import ABC, abstractmethod
import numpy as np
from electronTransportCode.ProjectUtils import tuple3d

# TODO: Find a more elegant way for users to pick arbitrary initial conditions
# TODO: Numba support. Remaining issues:
#   1) Numba doesn't support rng as class member. Use global rng in ProjectUtils.
#   2) No support for abstract base classes

class SimOptions(ABC):
    """Object which encapsulates initial condition and random number generator for Monte Carlo Simulation
    """
    def __init__(self, minEnergy: float, rngSeed: int = 12) -> None:
        self.minEnergy: float = minEnergy
        self.rng: np.random.Generator = np.random.default_rng(rngSeed)
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
        sign = self.rng.choice([-1, 1])
        sint = np.sqrt(1 - cost**2)*sign  # scatter left or right with equal probability

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


class LineSource(PointSource):
    def __init__(self, minEnergy: float, rngSeed: int, eSource: float, xmin: float, xmax: float) -> None:
        super().__init__(minEnergy, rngSeed, eSource)
        self.xmin = xmin
        self.xmax = xmax

    def initialPosition(self) -> tuple3d:
        """Source along the z-axis
        """
        return np.array((0.0, 0.0, self.rng.uniform(low=self.xmin, high=self.xmax, size=1)), dtype=float)
