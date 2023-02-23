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
    def __init__(self, minEnergy: float, Esource: float, xVariance: float, rngSeed: int = 12) -> None:
        super().__init__(minEnergy, rngSeed)
        self.Esource = Esource
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
        return self.Esource


class LineSourceSimulation(SimOptions):
    """Initial conditions for line source benchmark
    """
    def __init__(self, minEnergy: float, Esource: float, rngSeed: int = 12) -> None:
        super().__init__(minEnergy, rngSeed)
        self.Esource = Esource

    def initialDirection(self) -> tuple2d:
        """Uniformly distributed initial direction
        """
        theta = self.rng.uniform(low=0, high=2*np.pi)
        return np.array((np.cos(theta), np.sin(theta)))

    def initialPosition(self) -> tuple2d:
        """Initial position at origin
        """
        return np.zeros((2, ))

    def initialEnergy(self) -> float:
        """Constant particle energy source at self.Esource
        """
        return self.Esource
