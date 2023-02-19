from abc import ABC, abstractmethod
import numpy as np
from electronTransportCode.ProjectUtils import tuple2d


class SimOptions(ABC):
    """Object which encapsulates all simulation options for Monte Carlo simulation setings. E.g. number of particles, initial conditions, ....
    """
    def __init__(self, nbParticles: int, minEnergy: float, rngSeed: int = 12) -> None:
        self.nbParticles: int = nbParticles
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


class LineSourceSimulation(SimOptions):
    """Initial conditions for line source benchmark
    """
    def __init__(self, nbParticles: int, minEnergy: float, Esource: float, rngSeed: int = 12) -> None:
        super().__init__(nbParticles, minEnergy, rngSeed)
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
