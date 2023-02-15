import numpy as np
from MCEstimator import MCEstimator
from ParticleModel import ParticleModel
from abc import ABC, abstractmethod
from SimOptions import SimOptions
from utils import tuple2d
from SimulationDomain import SimulationDomain


class MCParticleTracer(ABC):
    def __init__(self, particle: ParticleModel, simOptions: SimOptions, simDomain: SimulationDomain) -> None:
        self.particle = particle
        self.simOptions = simOptions
        self.simDomain = simDomain
    
    def __call__(self, estimator: MCEstimator) -> None:
        """Execute Monte Carlo particle simulation of self.particle using self.simOptions in the simulation domain defined by self.simDomain. 
        Quantities of interest are estimated on-the-fly using estimator.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        for particleNb in range(self.simOptions.nbParticles):
            # simulate particle
            self.traceParticle(estimator)
            
        return None

    def energyLoss(self, Ekin: float, stepsize: float) -> float:
        """Compute energy along step using continuous slowing down approximation. Eq. 4.11.3 from EGSnrc manual, also used in GPUMCD.
        Approximation is second order accurate: O(DeltaE^2)

        Args:
            Ekin (float): Energy at the beginning of the step. Energy unit relative to electron rest energy.
            stepsize (float): [cm] path-length

        Returns:
            float: Energy loss DeltaE
        """
        Emid = Ekin + self.particle.evalStoppingPower(Ekin)*stepsize/2
        return self.particle.evalStoppingPower(Emid)*stepsize
            
    @abstractmethod
    def traceParticle(self, estimator: MCEstimator) -> None:
        pass
    
        
class AnalogParticleTracer(MCParticleTracer):
    def __init__(self, particle: ParticleModel, simOptions: SimOptions, simDomain: SimulationDomain) -> None:
        super().__init__(particle, simOptions, simDomain)
        
    def traceParticle(self, estimator: MCEstimator) -> None:
        """Step particle through simulation domain until its energy is below a threshold value. Estimator routine is called after each step for on-the-fly estimation. 

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        # Sample initial condition 
        pos: tuple2d = self.simOptions.initialPosition()
        vec: tuple2d = self.simOptions.initialDirection()
        energy: float = self.simOptions.initialEnergy()
        index: int = self.simDomain.returnIndex(pos)

        # Do type annotations for updated positions
        new_pos: tuple2d
        new_vec: tuple2d
        new_energy: float
        new_index: int

        # Step untill energy is smaller than threshold
        while energy >= self.simOptions.thresholdEnergy:
            new_pos, new_vec, new_energy, new_index = self.stepParticle(pos, vec, energy, index)
            estimator.updateEstimator((pos, new_pos), (vec, new_vec), (energy, new_energy), (index, new_index))
            pos = new_pos
            vec = new_vec
            energy = new_energy
            index = new_index
            
        return None
    
    def stepParticle(self, pos: tuple2d, vec: tuple2d, energy: float, index: int) -> tuple[tuple2d, tuple2d, float, int]:
        """Transport particle and apply event.
        Algorithm:
            # Sample step size
            #   1. collisional step size
            #   2. grid cell or domain edge step size

            # Apply step
            #   1. Change position
            #   2. Decrement energy

            # If next event is a collision
            #   1. Sample a new post-collisional direction
            #   2. Keep index

            # If next event is a boundary crossing
            #   1. Keep post-collisional direction
            #   2. Update index

            # If next event is a domain edge collision
            #   1. Set energy to zero
            #   2. Stop simulation by setting energy (weight) to zero


        Args:
            pos (tuple2d): particle position at start of method
            vec (tuple2d): particle orientation at start of method
            energy (float): particle energy at start of method
            index (int): particle position in self.simDomain at start of method

        Returns:
            tuple[tuple2d, tuple2d, float, int]: state of particle after being transported to new event location and having that event applied.
        """
        # Sample step size
        stepColl = self.particle.samplePathlength(energy)
        stepGeom, neighbourCellIndex = self.simDomain.getCellEdgeInformation(pos, vec, index)
        step = min(stepColl, stepGeom)
        
        # Apply step
        new_pos = pos + step*vec
        
        # Decrement energy along step
        deltaE = self.energyLoss(energy, step)
        new_energy = energy - deltaE
        
        # Select event
        if stepColl < stepGeom:  # Next event is collision
            new_vec: tuple2d = np.zeros_like(vec)
            new_index = index
            cost = self.particle.sampleAngle(new_energy)  # anisotropic scattering angle (mu)
            sign = self.simOptions.rng.choice([-1, 1])
            sint = np.sqrt(1 - cost**2)*sign  # scatter left or right with equal probability
            new_vec[0] = vec[0]*cost - vec[1]*sint
            new_vec[1] = vec[0]*sint + vec[1]*cost
            
        else:  # Next event is grid cell crossing
            new_index = index
            new_vec = vec
            if neighbourCellIndex == -1:  # Next event is domain edge crossing
                new_energy = 0
                
        return new_pos, new_vec, new_energy, new_index
                    