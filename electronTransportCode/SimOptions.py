from abc import ABC, abstractmethod
import numpy as np
from electronTransportCode.ProjectUtils import tuple2d

# TODO: Find a more elegant way for users to pick arbitrary initial conditions

class SimOptions(ABC):
    """Object which encapsulates initial condition and random number generator for Monte Carlo Simulation
    """
    def __init__(self, minEnergy: float, rngSeed: int = 12) -> None:
        self.minEnergy: float = minEnergy
        self.rng: np.random.Generator = np.random.default_rng(rngSeed)
        self.DEPOSIT_REMAINDING_E_LOCALLY = True

    @abstractmethod
    def initialDirection(self) -> tuple2d:
        """Sample initial direction of particle
        """

    @abstractmethod
    def initialPosition(self) -> tuple2d:
        """Sample initial position of particle
        """

    @abstractmethod
    def initialEnergy(self) -> float:
        """Sample initial energy of particle
        """


class WaterPhantomSimulation(SimOptions):
    """Initial conditions for water phantom experiment
    """
    def __init__(self, minEnergy: float, eSource: float, xVariance: float, rngSeed: int = 12) -> None:
        super().__init__(minEnergy, rngSeed)
        self.eSource = eSource
        self.xVariance = xVariance

    def initialDirection(self) -> tuple2d:
        """Particle moving to the right
        """
        return np.array((1.0, 1.0))*np.sqrt(2)/2

    def initialPosition(self) -> tuple2d:
        """Initial position at origin
        """
        xSample = self.rng.normal(scale=np.sqrt(self.xVariance))
        ySample = self.rng.normal(scale=np.sqrt(self.xVariance))
        return np.array((xSample, ySample))

    def initialEnergy(self) -> float:
        """Constant particle energy source at self.Esource
        """
        return self.eSource


class PointSourceSimulation(SimOptions):
    """Initial conditions for point source benchmark
    """
    def __init__(self, minEnergy: float, rngSeed: int, eSource: float) -> None:
        super().__init__(minEnergy, rngSeed)
        self.eSource = eSource

    def initialDirection(self) -> tuple2d:
        """Uniformly distributed initial direction
        """
        # isotropic angle
        # theta = self.rng.uniform(low=0, high=2*np.pi)
        # return np.array((np.cos(theta), np.sin(theta)))

        # isotropic cos(theta)
        cost = self.rng.uniform(low=-1, high=1)
        sign = self.rng.choice([-1, 1])
        sint = np.sqrt(1 - cost**2)*sign  # scatter left or right with equal probability
        return np.array((cost, sint), dtype=float)

    def initialPosition(self) -> tuple2d:
        """Initial position at origin
        """
        return np.zeros((2, ))

    def initialEnergy(self) -> float:
        """Constant particle energy source at self.Esource
        """
        return self.eSource
