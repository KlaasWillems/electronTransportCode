from abc import ABC, abstractmethod
import numpy as np
from electronTransportCode.utils import tuple2d


class SimOptions(ABC):
    """Object which encapsulates all simulation options for Monte Carlo simulation setings. E.g. number of particles, initial conditions, ....
    """
    def __init__(self, nbParticles: int, thresholdEnergy: float) -> None:
        self.nbParticles: int = nbParticles
        self.thresholdEnergy: float = thresholdEnergy
        
    @abstractmethod
    def initialDirection(self) -> tuple2d:
        # Result must be a unit vector
        pass
    
    @abstractmethod
    def initialPosition(self) -> tuple2d:
        pass
    
    @abstractmethod
    def initialEnergy(self) -> float:
        pass
    
