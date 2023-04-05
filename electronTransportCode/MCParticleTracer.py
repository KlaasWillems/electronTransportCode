from abc import abstractmethod, ABC
import time
from mpi4py import MPI
from typing import Union, Tuple, Optional
import numpy as np
import pickle
from electronTransportCode.MCEstimator import MCEstimator
from electronTransportCode.ParticleModel import ParticleModel
from electronTransportCode.SimOptions import SimOptions
from electronTransportCode.ProjectUtils import tuple3d, tuple3d
from electronTransportCode.SimulationDomain import SimulationDomain


class ParticleTracer(ABC):
    """General Monte Carlo particle tracer object for radiation therapy. Particle move in a 3D domain however,
    estimation and grid cell crossing are only supported in 2D (yz-plane).
    """
    def __init__(self, simOptions: SimOptions, simDomain: SimulationDomain, particle: Optional[ParticleModel]) -> None:
        """
        Args:
            particle (ParticleModel): Particle to transport
            simOptions (SimOptions): Initial conditions & simulation parameters
            simDomain (SimulationDomain): Simulation domain
        """
        self.particle = particle
        self.simOptions = simOptions
        self.simDomain = simDomain
        if particle is not None:
            assert self.particle is not None
            self.particle.setGenerator(self.simOptions.rng)  # Have all random numbers in simulation be generated by the same generator object

    def runMultiProc(self, nbParticles: int, estimators: Union[MCEstimator, tuple[MCEstimator, ...]], file: Optional[str], SEED: int = 12, root: int = 0, particle: Optional[ParticleModel] = None, logAmount: int = np.iinfo(int).max, verbose: bool = True) -> None:
        """Run simulation in parallel using MPI.

        Args:
            nbParticles (int): Total amount of particles
            estimators (Union[MCEstimator, tuple[MCEstimator, ...]]): List of estimators. These estimators are gathered and written to a file after simulation.
            file (Optional[str]): File to write estimators at.
            SEED (int, optional): RNG seed. Each process is assigned a different seed. RNGs of self.simOptions and self.particle are also reassigned. Defaults to 12.
            root (int, optional): Process which gathers all estimators, combines them and writes them to a file. Defaults to 0.
            particle (Optional[ParticleModel], optional): The particle to simulate. Defaults to None.
            logAmount (int, optional): After 'logAmount' of particles, a message is printed with the progress of the simulation. Defaults to np.iinfo(int).max.
            verbose (bool): Turn on non-timing related printing. Defaults to True.
        """

        if isinstance(estimators, MCEstimator):  # estimator is now a list of estimators
            estimatorList: tuple[MCEstimator, ...] = (estimators, )
        else:
            estimatorList = estimators

        if particle is not None:
            self.particle = particle
            self.particle.rng = self.simOptions.rng
        assert self.particle is not None

        myrank = MPI.COMM_WORLD.Get_rank()
        nproc = MPI.COMM_WORLD.Get_size()

        # Amount of particles to simulate per processor
        particles_per_proc = int(nbParticles/nproc)

        # Reset seed for multiproc use
        self.simOptions.rng = np.random.default_rng(SEED+myrank)
        self.particle.setGenerator(self.simOptions.rng)

        # Run simulation
        if verbose:
            print(f'Proc {myrank} starting simulation of {particles_per_proc} particles.')
        self.__call__(particles_per_proc, estimatorList, logAmount=logAmount)

        # Gather results
        for estimator in estimatorList:
            estimator.combineEstimators(particles_per_proc, root=0)

        # store results
        if myrank == root and file is not None:
            pickle.dump(estimatorList, open(file, 'wb'))

    def __call__(self, nbParticles: int, estimators: Union[MCEstimator, tuple[MCEstimator, ...]], logAmount: int = 1000, particle: Optional[ParticleModel] = None ) -> Union[MCEstimator, tuple[MCEstimator, ...]]:
        """Execute Monte Carlo particle simulation of self.particle using self.simOptions in the simulation domain defined by self.simDomain.
        Quantities of interest are estimated on-the-fly using estimator.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest
            logAmount (int) : Print time every logAmount seconds
            particle (Optional[ParticleModel]) : If particle is not None, make particle self.particle and simulate it.

        """
        self.resetAnalytics()
        myrank = MPI.COMM_WORLD.Get_rank()

        if particle is not None:
            self.particle = particle
            self.particle.rng = self.simOptions.rng
        assert self.particle is not None

        t1 = time.perf_counter()
        for particleID in range(1, nbParticles+1):
            self.traceParticle(estimators)

            if particleID % logAmount == 0:  # every 1000 particles
                t2 = time.perf_counter()
                print(f'Process: {myrank}, type: {self.__class__.__name__}. Last {logAmount} particles took {t2-t1} seconds. {100*particleID/nbParticles}% completed.')
                t1 = time.perf_counter()

        return estimators

    def energyLoss(self, Ekin: float, pos3d: tuple3d, stepsize: float, index: int) -> float:
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
        assert self.particle is not None
        Emid = Ekin + self.particle.evalStoppingPower(Ekin, pos3d, self.simDomain.getMaterial(index))*stepsize/2
        assert Emid > 0, f'{Emid=}'
        return self.particle.evalStoppingPower(Emid, pos3d, self.simDomain.getMaterial(index))*stepsize

    def stepParticleAnalog(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int) -> tuple[tuple3d, tuple3d, float, int, float, bool]:
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
            kin_stepped (bool): if there was a collision event or not

        Returns:
            tuple[tuple2d, tuple2d, float, int, float]: state of particle after being transported to new event location and having that event applied. Final return value is the step size.
        """
        assert self.particle is not None
        kin_stepped: bool = False

        # Sample step size
        stepColl = self.particle.samplePathlength(energy, pos3d, self.simDomain.getMaterial(index))
        stepGeom, domainEdge, new_pos3d_geom = self.simDomain.getCellEdgeInformation(pos3d, vec3d, index)
        step = min(stepColl, stepGeom)

        # Apply step
        if step == stepGeom:
            new_pos3d = new_pos3d_geom
        else:
            kin_stepped = True
            new_pos3d = pos3d + step*vec3d

        # Decrement energy along step
        deltaE = self.energyLoss(energy, pos3d, step, index)
        new_energy = energy - deltaE

        if new_energy < self.simOptions.minEnergy:  # return without sampling a new angle and such
            # linearly back up such that stepsize is consistent with energy loss
            step_lin = step*(energy - self.simOptions.minEnergy)/deltaE
            new_pos3d = pos3d + step_lin*vec3d
            return new_pos3d, vec3d, self.simOptions.minEnergy, index, step_lin, kin_stepped

        # Select event
        if stepColl < stepGeom:  # Next event is collision
            new_vec3d: tuple3d = np.array((0.0, 0.0, 0.0), dtype=float)
            new_index = index

            new_vec3d = self.particle.sampleNewVec(new_pos3d, vec3d, new_energy, self.simDomain.getMaterial(index))
            return new_pos3d, new_vec3d, new_energy, new_index, step, kin_stepped

        else:  # Next event is grid cell crossing
            new_vec3d = vec3d
            new_index = self.simDomain.getIndexPath(new_pos3d, new_vec3d)
            if domainEdge:  # Next event is domain edge crossing
                new_energy = 0

            return new_pos3d, new_vec3d, new_energy, new_index, step, kin_stepped

    @abstractmethod
    def traceParticle(self, estimators: Union[MCEstimator, tuple[MCEstimator, ...]]) -> None:
        pass

    @abstractmethod
    def resetAnalytics(self) -> None:
        pass

    @abstractmethod
    def updateAnalytics(self, *args, **kwargs) -> None:
        pass


class AnalogParticleTracer(ParticleTracer):
    """Analog particle tracer which executes each scattering collision explicitly.
    """

    def traceParticle(self, estimators: Union[MCEstimator, tuple[MCEstimator, ...]]) -> None:
        """Step particle through simulation domain until its energy is below a threshold value. Estimator routine is called after each step for on-the-fly estimation.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        if isinstance(estimators, MCEstimator):  # estimator is now a list of estimators
            estimatorList: tuple[MCEstimator, ...] = (estimators, )
        else:
            estimatorList = estimators

        # Sample initial condition in 2D
        pos3d: tuple3d = self.simOptions.initialPosition()
        vec3d: tuple3d = self.simOptions.initialDirection()
        energy: float = self.simOptions.initialEnergy()
        index: int = self.simDomain.getIndexPath(pos3d, vec3d)

        loopbool: bool = True

        # Do type annotations for updated positions
        new_pos3d: tuple3d
        new_vec3d: tuple3d
        new_energy: float
        new_index: int

        counter = 0
        # Step until energy is smaller than threshold
        while loopbool:
            assert energy > self.simOptions.minEnergy, f'{energy=}'
            new_pos3d, new_vec3d, new_energy, new_index, step_kin, kin_stepped = self.stepParticleAnalog(pos3d, vec3d, energy, index)

            if new_energy <= self.simOptions.minEnergy: # have estimator deposit all remaining energy
                if self.simOptions.DEPOSIT_REMAINDING_E_LOCALLY:
                    new_energy = 0
                loopbool = False  # make this the last iterations

            for estimator in estimatorList:
                estimator.updateEstimator((pos3d, new_pos3d), (vec3d, new_vec3d), (energy, new_energy), index, step_kin)

            pos3d = new_pos3d
            vec3d = new_vec3d
            energy = new_energy
            index = new_index

            if kin_stepped: counter += 1

        self.updateAnalytics(counter)

    def resetAnalytics(self) -> None:
        self.averageNbCollisions = 0.0
        self.particleIndex: int = 0

    def updateAnalytics(self, NbCollisions: int) -> None:
        self.averageNbCollisions = (self.particleIndex*self.averageNbCollisions + NbCollisions)/(self.particleIndex+1)
        self.particleIndex += 1


class KDParticleTracer(ParticleTracer, ABC):
    """Implements kinetic-diffusion Monte Carlo using a mean and variance.
    """

    def __init__(self, simOptions: SimOptions, simDomain: SimulationDomain, particle: Optional[ParticleModel], dS: Optional[float] = None) -> None:
        super().__init__(simOptions, simDomain, particle)
        self.dS = dS  # KDMC stepsize parameter

    def resetAnalytics(self) -> None:
        self.particleIndex: int = 0
        self.AvgNbAnalogCollisions: float = 0.0
        self.AvgNbDiffCollisions: float = 0.0

    def updateAnalytics(self, NbAnalogCollisions: int, NbDiffCollisions: int) -> None:
        self.AvgNbAnalogCollisions = (self.particleIndex*self.AvgNbAnalogCollisions + NbAnalogCollisions)/(self.particleIndex+1)
        self.AvgNbDiffCollisions = (self.particleIndex*self.AvgNbDiffCollisions + NbDiffCollisions)/(self.particleIndex+1)
        self.particleIndex += 1

    def traceParticle(self, estimators: Union[MCEstimator, tuple[MCEstimator, ...]]) -> None:
        """Step particle through simulation domain until its energy is below a threshold value. Estimator routine is called after each step for on-the-fly estimation.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        assert self.dS is not None
        assert self.particle is not None
        if isinstance(estimators, MCEstimator):  # estimator is now a list of estimators
            estimatorList: tuple[MCEstimator, ...] = (estimators, )
        else:
            estimatorList = estimators

        # Sample initial condition in 2D
        pos3d: tuple3d = self.simOptions.initialPosition()
        vec3d: tuple3d = self.simOptions.initialDirection()
        energy: float = self.simOptions.initialEnergy()
        index: int = self.simDomain.getIndexPath(pos3d, vec3d)

        loopbool: bool = True

        # Do type annotations for updated positions
        kin_pos3d: tuple3d; diff_pos3d: tuple3d
        kin_vec3d: tuple3d; diff_vec3d: tuple3d
        kin_energy: float; diff_energy: float
        kin_index: int; diff_index: int

        # Couter kinetic and diffusive steps
        kin_counter: int = 0
        diff_counter: int = 0

        # Step until energy is smaller than threshold
        while loopbool:
            assert energy > self.simOptions.minEnergy, f'{energy=}'

            # do analog step
            kin_pos3d, kin_vec3d, kin_energy, kin_index, step_kin, kin_stepped = self.stepParticleAnalog(pos3d, vec3d, energy, index)
            if kin_stepped: kin_counter += 1

            # If no energy left
            if kin_energy <= self.simOptions.minEnergy: # have estimator deposit all remaining energy
                if self.simOptions.DEPOSIT_REMAINDING_E_LOCALLY:
                    kin_energy = 0
                loopbool = False  # make this the last iteration

            # Score QOIs of kinetic step
            for estimator in estimatorList:
                estimator.updateEstimator((pos3d, kin_pos3d), (vec3d, kin_vec3d), (energy, kin_energy), kin_index, step_kin)

            if kin_energy > self.simOptions.minEnergy:  # Do diffusive step if energy left
                step_diff = self.dS - (step_kin % self.dS)

                diff_pos3d, diff_vec3d, diff_energy, diff_index, diff_stepped = self.stepParticleDiffusive(kin_pos3d, kin_vec3d, kin_energy, kin_index, step_diff)
                if diff_stepped: diff_counter += 1

                # Final veloctiy and index must remain the same
                assert np.all(kin_vec3d == diff_vec3d)
                assert diff_index == kin_index

                # If energy left
                if diff_energy <= self.simOptions.minEnergy: # have estimator deposit all remaining energy
                    if self.simOptions.DEPOSIT_REMAINDING_E_LOCALLY:
                        diff_energy = 0
                    loopbool = False  # make this the last iteration

                # Score QOIs
                for estimator in estimatorList:
                    estimator.updateEstimator((kin_pos3d, diff_pos3d), (kin_vec3d, diff_vec3d), (kin_energy, diff_energy), kin_index, step_diff)

                # set final state
                pos3d = diff_pos3d
                vec3d = diff_vec3d
                energy = diff_energy
                index = diff_index

        self.updateAnalytics(kin_counter, diff_counter)

    def stepParticleDiffusive(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> tuple[tuple3d, tuple3d, float, int, bool]:
        """Apply diffusive motion to particle

        Args:
            pos3d (tuple3d): Position of particle after kinetic step
            vec3d (tuple3d): final velocity on which the diffusive moment is conditioned
            energy (float): energy of particle after kinetic step
            index (int): position of particle in estimator grid
            stepsize (float): remaing part of step for diffusive motion

        Returns:
            tuple[tuple3d, tuple3d, float, int, bool]: state of particle after diffusive step. Final boolean indicates if diffusive step was taken or not (due to out of bounds are steps diffusive step can be skipped).
        """
        assert self.particle is not None

        # Get mean and variance of Omega distribution
        mu_omega, _ = self.particle.getOmegaMoments(pos3d)

        # Fix intermediate position
        x_int = pos3d + mu_omega*stepsize/2

        # Get advection and diffusion coefficient
        A_coeff, D_coeff = self.advectionDiffusionCoeff(x_int, vec3d, stepsize)

        # Apply diffusive step
        xi = self.simOptions.rng.normal(size=(3, ))
        new_pos3d = pos3d + A_coeff*stepsize + np.sqrt(2*D_coeff*stepsize)*xi

        # Find equivalent kinetic step
        pos_delta = new_pos3d - pos3d
        equi_step = np.sqrt(pos_delta[0]**2 + pos_delta[1]**2 + pos_delta[2]**2)
        equi_vec = pos_delta/equi_step

        # Figure out if the particle when out of the grid cell.
        stepGeom, _, _ = self.simDomain.getCellEdgeInformation(pos3d, equi_vec, index)
        if equi_step < stepGeom: # diffusive step did not exceed boundaries

            # Decrement energy along step
            deltaE = self.energyLoss(energy, pos3d, stepsize, index)
            new_energy = energy - deltaE

            # Figure out if the particle did not exceed the amount of energy it had left
            if new_energy < self.simOptions.minEnergy:
                return pos3d, vec3d, energy, index, False  # don't move particle, return old state
            else:
                return new_pos3d, vec3d, new_energy, index, True

        else: # diffusive step exceeded cell boundaries
            return pos3d, vec3d, energy, index, False  # don't move particle, return old state

    @abstractmethod
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        pass

class KDMC(KDParticleTracer):
    """Implements kinetic-diffusion Monte Carlo using the mean and variance of kinetic motion conditioned on the final velocity.
    """
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        """Return advection and diffusion coefficient

        Args:
            pos3d (tuple3d): Intermediate position x double prime
            vec3d (tuple3d): final direction of travel on which the mean and variance are conditioned
            stepsize (float): remaining 'time' for the diffusive movement

        Returns:
            Tuple[tuple3d, tuple3d]: Advection and diffusion coefficient
        """
        assert self.particle is not None
        mu_omega, var_omega = self.particle.getOmegaMoments(pos3d)
        Sigma_s = self.particle.getScatteringRate(pos3d)  # TODO: add material and energy dependence (also with the derivative)

        # intermediate results
        stepsizeSigma2 = stepsize*(Sigma_s**2)
        sigmaStepsize = Sigma_s*stepsize
        exp1: float = np.exp(-sigmaStepsize)
        exp2: float = np.exp(-2*sigmaStepsize)
        vec_mean_dev = vec3d - mu_omega
        vec_mean_dev2 = vec_mean_dev**2

        # Heterogeneity correction
        het1 = 0.5*vec_mean_dev2*(exp2 + 2*sigmaStepsize*exp1 - 1.0)/stepsizeSigma2
        het2 = var_omega*(2*exp1 + sigmaStepsize + sigmaStepsize*exp1 - 2.0)/stepsizeSigma2
        het3 = var_omega*(sigmaStepsize*exp1 - 1.0 + exp1)/Sigma_s
        het4 = vec_mean_dev2*(sigmaStepsize*exp1 - exp1 + exp2)/Sigma_s
        het_correction = het1 - het2 - het3 + het4

        # Mean
        dRdx = self.particle.getDScatteringRate(pos3d)
        mean: tuple3d = mu_omega + (1 - exp1)*vec_mean_dev/sigmaStepsize + het_correction*dRdx

        # Variance
        temp1 = 1.0 - exp1*2*sigmaStepsize - exp2  # Due to catastrophic cancellation, this thing can become negative. In this case, put it to zero.
        if temp1 < 0: temp1 = 0
        var_term1 = vec_mean_dev2*temp1/(2*stepsizeSigma2)
        temp2 = 2*exp1 + sigmaStepsize + sigmaStepsize*exp1 - 2.0  # Due to catastrophic cancellation, this thing can become negative. In this case, put it to zero.
        if temp2 < 0: temp2 = 0
        var_term2 = var_omega*temp2/stepsizeSigma2

        return mean, var_term1 + var_term2


class KDsMC(KDParticleTracer):
    """Implements KDMC using a mean and variance that doesn't incorporate the correlation between multiple time steps. The 's' stands for 'simple'.
    """
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        raise NotImplementedError

class KDDLMC(KDParticleTracer):
    """Implements KDMC using the advection and diffusion coefficients from the diffusion limit.
    """
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        raise NotImplementedError
