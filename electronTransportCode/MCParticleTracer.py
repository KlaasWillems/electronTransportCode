from abc import ABC, abstractmethod
import time
import numpy as np
from electronTransportCode.MCEstimator import MCEstimator
from electronTransportCode.ParticleModel import ParticleModel
from electronTransportCode.SimOptions import SimOptions
from electronTransportCode.ProjectUtils import tuple2d
from electronTransportCode.SimulationDomain import SimulationDomain


class MCParticleTracer(ABC):
    """General Monte Carlo particle tracer object for radiation therapy
    """
    def __init__(self, particle: ParticleModel, simOptions: SimOptions, simDomain: SimulationDomain) -> None:
        self.particle = particle
        self.simOptions = simOptions
        self.simDomain = simDomain
        self.particle.setGenerator(self.simOptions.rng)  # Have all random numbers in simulation be generated by the same generator object
        self.averageNbCollisions = 0.0

    def __call__(self, nbParticles: int, estimator: MCEstimator) -> None:
        """Execute Monte Carlo particle simulation of self.particle using self.simOptions in the simulation domain defined by self.simDomain.
        Quantities of interest are estimated on-the-fly using estimator.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        t1 = time.perf_counter()
        for particle in range(1, nbParticles+1):
            nb_events = self.traceParticle(estimator)
            self.averageNbCollisions += (nb_events - self.averageNbCollisions)/particle

            if particle % 1000 == 0:  # every 1000 particles
                t2 = time.perf_counter()
                print(f'Last 1000 particles took {t2-t1} seconds')
                t1 = time.perf_counter()

        return None

    def energyLoss(self, Ekin: float, stepsize: float, index: int) -> float:
        """Compute energy along step using continuous slowing down approximation. Eq. 4.11.3 from EGSnrc manual, also used in GPUMCD.
        Approximation is second order accurate: O(DeltaE^2)

        Args:
            Ekin (float): Energy at the beginning of the step. Energy unit relative to electron rest energy.
            stepsize (float): [cm] path-length

        Returns:
            float: Energy loss DeltaE
        """
        assert Ekin > 0, f'{Ekin=}'
        assert stepsize > 0, f'{stepsize=}'
        Emid = Ekin + self.particle.evalStoppingPower(Ekin, self.simDomain.getMaterial(index))*stepsize/2
        assert Emid > 0, f'{Emid=}'
        return self.particle.evalStoppingPower(Emid, self.simDomain.getMaterial(index))*stepsize

    @abstractmethod
    def traceParticle(self, estimator: MCEstimator) -> int:
        """Simulate one particle
        """


class AnalogParticleTracer(MCParticleTracer):
    """Analog particle tracing algorithm
    """
    def traceParticle(self, estimator: MCEstimator) -> int:
        """Step particle through simulation domain until its energy is below a threshold value. Estimator routine is called after each step for on-the-fly estimation.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        # Sample initial condition
        pos: tuple2d = self.simOptions.initialPosition()
        vec: tuple2d = self.simOptions.initialDirection()
        energy: float = self.simOptions.initialEnergy()
        index: int = self.simDomain.getIndexPath(pos, vec)

        loopbool: bool = True

        # Do type annotations for updated positions
        new_pos: tuple2d
        new_vec: tuple2d
        new_energy: float
        new_index: int

        counter = 0
        # Step until energy is smaller than threshold
        while loopbool:
            assert energy > self.simOptions.minEnergy, f'{energy=}'
            new_pos, new_vec, new_energy, new_index = self.stepParticle(pos, vec, energy, index)
            if new_energy <= self.simOptions.minEnergy:
                if self.simOptions.DEPOSIT_REMAINDING_E_LOCALLY:
                    new_energy = 0  # have estimator deposit all energy
                loopbool = False  # make this the last iterations
            estimator.updateEstimator((pos, new_pos), (vec, new_vec), (energy, new_energy), index)
            pos = new_pos
            vec = new_vec
            energy = new_energy
            index = new_index

            # Logging
            counter += 1
            # if counter % 25000 == 0:
            #     print(energy, counter)

        return counter

    def stepParticle(self, pos: tuple2d, vec: tuple2d, energy: float, index: int) -> tuple[tuple2d, tuple2d, float, int]:
        """Transport particle and apply event.
        Algorithm:
            - Sample step size
                - collisional step size
                - grid cell or domain edge step size

            - Apply step
                - Change position
                - Decrement energy

            - If next event is a collision
                - Sample a new post-collisional direction
                - Keep index

            - If next event is a boundary crossing
                - Keep post-collisional direction
                - Update index

            - If next event is a domain edge collision
                - Set energy to zero
                - Stop simulation by setting energy (weight) to zero

        Args:
            pos (tuple2d): particle position at start of method
            vec (tuple2d): particle orientation at start of method
            energy (float): particle energy at start of method
            index (int): particle position in self.simDomain at start of method

        Returns:
            tuple[tuple2d, tuple2d, float, int]: state of particle after being transported to new event location and having that event applied.
        """

        # Sample step size
        stepColl = self.particle.samplePathlength(energy, self.simDomain.getMaterial(index))
        stepGeom, domainEdge, new_pos_geom = self.simDomain.getCellEdgeInformation(pos, vec, index)
        step = min(stepColl, stepGeom)

        # Apply step
        if step == stepGeom:
            new_pos = new_pos_geom
        else:
            new_pos = pos + step*vec

        # Decrement energy along step
        deltaE = self.energyLoss(energy, step, index)
        new_energy = energy - deltaE

        if new_energy < self.simOptions.minEnergy:  # return without sampling a new angle and such
            # linearly back up such that stepsize is consistent with energy loss
            new_pos = pos + step*vec*(energy - self.simOptions.minEnergy)/deltaE
            return new_pos, vec, self.simOptions.minEnergy, index

        # Select event
        if stepColl < stepGeom:  # Next event is collision
            new_vec: tuple2d = np.zeros_like(vec, dtype=float)
            new_index = index
            cost = self.particle.sampleAngle(new_energy, self.simDomain.getMaterial(index))  # anisotropic scattering angle (mu)
            sign = self.simOptions.rng.choice([-1, 1])
            sint = np.sqrt(1 - cost**2)*sign  # scatter left or right with equal probability
            new_vec[0] = vec[0]*cost - vec[1]*sint
            new_vec[1] = vec[0]*sint + vec[1]*cost
            new_vec = new_vec/np.linalg.norm(new_vec)  # normalized for security

        else:  # Next event is grid cell crossing
            new_vec = vec
            new_index = self.simDomain.getIndexPath(new_pos, new_vec)
            if domainEdge:  # Next event is domain edge crossing
                new_energy = 0

        return new_pos, new_vec, new_energy, new_index
