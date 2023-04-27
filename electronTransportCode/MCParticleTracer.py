from abc import abstractmethod, ABC
import time
import math
from mpi4py import MPI
from typing import Union, Tuple, Optional
import numpy as np
import pickle

from electronTransportCode.MCEstimator import MCEstimator
from electronTransportCode.ParticleModel import ParticleModel
from electronTransportCode.SimOptions import SimOptions
from electronTransportCode.ProjectUtils import ERE, tuple3d, tuple3d
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
            print(f'Proc: {myrank}, type: {self.__class__.__name__}. Starting simulation of {particles_per_proc} particles.')

        self.__call__(particles_per_proc, estimatorList, logAmount=logAmount)

        # Gather results
        for estimator in estimatorList:
            estimator.combineEstimators(particles_per_proc, root=0)

        # store results
        if myrank == root and file is not None:
            pickle.dump(estimatorList, open(file, 'wb'))

    def __call__(self, nbParticles: int, estimators: Union[MCEstimator, tuple[MCEstimator, ...]], logAmount: int = np.iinfo(int).max, particle: Optional[ParticleModel] = None ) -> Union[MCEstimator, tuple[MCEstimator, ...]]:
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
                print(f'Proc: {myrank}, type: {self.__class__.__name__}. Last {logAmount} particles took {round(t2-t1, 4)} seconds. {round(100*particleID/nbParticles, 3)}% completed.')
                t1 = time.perf_counter()

        return estimators

    def scatterParticle(self, cost: float, phi: float, vec3d: tuple3d, new_direction: bool) -> tuple3d:
        """Scatter a particle across polar angle (theta) and azimuthatl scattering angle phi

        Args:
            cost (float): Cosine of polar scattering angle
            phi (float): Azimuthal scattering angle in radians
            vec3d (tuple3d): Direction of travel

        Returns:
            tuple3d: Direction of travel after scattering cost and phi with respect to vec3d
        """
        sint = math.sqrt(1 - cost**2)
        cosphi = math.cos(phi)
        sinphi = math.sin(phi)

        if new_direction:
            x = sint*cosphi
            y = sint*sinphi
            z = cost
        else:
            # Rotation matrices (See penelope documentation eq. 1.131)
            if math.isclose(abs(vec3d[2]), 1.0, rel_tol=1e-14):  # indeterminate case
                sign = math.copysign(1.0, vec3d[2])
                x = sign*sint*cosphi
                y = sign*sint*sinphi
                z = sign*cost
            else:
                tempVar = math.sqrt(1-vec3d[2]**2)
                tempVar2 = vec3d[2]*cosphi
                x = vec3d[0]*cost + sint*(vec3d[0]*tempVar2 - vec3d[1]*sinphi)/tempVar
                y = vec3d[1]*cost + sint*(vec3d[1]*tempVar2 + vec3d[0]*sinphi)/tempVar
                z = vec3d[2]*cost - tempVar*sint*cosphi

        # normalized for security
        norm = math.sqrt(x**2 + y**2 + z**2)
        return np.array((x/norm, y/norm, z/norm), dtype=float)

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
            new_pos3d = pos3d + step*vec3d  # type: ignore

        # Decrement energy along step
        material = self.simDomain.getMaterial(index)
        deltaE = self.particle.energyLoss(energy, pos3d, step, material)
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

            mu, phi, new_direction = self.particle.sampleScatteringAngles(new_energy, material)
            new_vec3d = self.scatterParticle(mu, phi, vec3d, new_direction)
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
        self.AvgNbFalseDiffCollisions: float = 0.0

    def updateAnalytics(self, NbAnalogCollisions: int, NbDiffCollisions: int, NbFalseDiffCollisions) -> None:
        self.AvgNbAnalogCollisions = (NbAnalogCollisions - self.AvgNbAnalogCollisions)/(self.particleIndex+1) + self.AvgNbAnalogCollisions
        self.AvgNbDiffCollisions = (NbDiffCollisions - self.AvgNbDiffCollisions)/(self.particleIndex+1) + self.AvgNbDiffCollisions
        self.AvgNbFalseDiffCollisions = (NbFalseDiffCollisions - self.AvgNbFalseDiffCollisions)/(self.particleIndex+1) + self.AvgNbFalseDiffCollisions
        self.particleIndex += 1

    def pickStepSize(self, kin_pos3d: tuple3d, kin_energy: float, kin_index: int, step_kin: float, N: int = 4) -> Tuple[float, float]:
        """Pick a stepsize for the diffusive step. Normally diffusive step is such that after a kinetic step, the remaining part of dS is done diffusively. When this stepsize results in an energy loss below self.simOptions.minEnergy, a smaller stepsize is chosen such that minEnergy is reached.

        Args:
            kin_pos3d (tuple3d): Position of particle
            kin_energy (float): Energy of particle after kinetic step
            kin_index (int): Index of particle after kinetic step
            step_kin (float): Size of kinetic step
            N (int, optional): Amount of interpolation points for trapezoidal rule. Defaults to 4.

        Returns:
            Tuple[float, float]: Diffusive step size and new energy lost during diffusive step
        """
        assert self.particle is not None
        assert self.dS is not None

        material = self.simDomain.getMaterial(kin_index)
        step_diff = self.dS - (step_kin % self.dS)
        deltaE = self.particle.energyLoss(kin_energy, kin_pos3d, step_diff, material)
        new_energy = kin_energy - deltaE

        if new_energy >= self.simOptions.minEnergy:  # stepsize did not exceed energy
            return step_diff, new_energy
        else:  # stepsize exceeded remaining energy: Approximate stepsize to reach minEnergy using trapezoidal rule
            Erange = np.linspace(kin_energy, self.simOptions.minEnergy, N)
            sps = [self.particle.evalStoppingPower(e, kin_pos3d, material) for e in Erange]
            dss = [2*(Erange[i] - Erange[i+1])/(sps[i] + sps[i+1]) for i in range(N-1)]
            return sum(dss), self.simOptions.minEnergy

    def traceParticle(self, estimators: Union[MCEstimator, tuple[MCEstimator, ...]]) -> None:
        """Step particle through simulation domain until its energy is below a threshold value. Estimator routine is called after each step for on-the-fly estimation.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        assert self.dS is not None
        assert self.particle is not None
        if isinstance(estimators, MCEstimator):  # estimator is now a list of estimators
            estimatorList: Tuple[MCEstimator, ...] = (estimators, )
        else:
            estimatorList = estimators

        # Sample initial condition in 2D
        pos3d: tuple3d = self.simOptions.initialPosition()
        vec3d: tuple3d = self.simOptions.initialDirection()
        energy: float = self.simOptions.initialEnergy()
        index: int = self.simDomain.getIndexPath(pos3d, vec3d)

        loopbool: bool = True

        # Couter kinetic and diffusive steps
        kin_counter: int = 0
        diff_counter: int = 0
        false_diff_counter: int = 0

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

            if loopbool:  # Do diffusive step if there is energy left

                step_diff1, diff_energy1 = self.pickStepSize(kin_pos3d, kin_energy, kin_index, step_kin)

                diff_pos3d, equi_vec, diff_index, diff_stepped, diff_step_track,  = self.stepParticleDiffusive(kin_pos3d, kin_vec3d, kin_energy, kin_index, step_diff1)
                if diff_stepped:  # Step was taken: do estimation
                    assert equi_vec is not None

                    diff_counter += 1
                    diff_energy = diff_energy1

                    # If energy left
                    if diff_energy <= self.simOptions.minEnergy: # have estimator deposit all remaining energy
                        if self.simOptions.DEPOSIT_REMAINDING_E_LOCALLY:
                            diff_energy = 0
                        loopbool = False  # make this the last iteration

                    diff_vec3d = kin_vec3d  # For conditioning on final velocity in KDMC

                    # Divide energy loss evenly over all crossed cells based on stepsize
                    dE = kin_energy - diff_energy
                    steps = np.array([x[1] for x in diff_step_track], dtype=float)
                    eHistory = kin_energy - np.cumsum(steps*dE/steps.sum())  # type: ignore
                    eHistory[-1] = diff_energy

                    # Score QOIs: score each track in each cell that was crossed seperatly
                    startpos_it = kin_pos3d
                    startvec_it = kin_vec3d
                    endvec_it = equi_vec
                    startEnergy_it = kin_energy
                    for it, tuples in enumerate(diff_step_track):
                        if it == len(diff_step_track)-1:
                            endvec_it = diff_vec3d
                        index_it, stepsize_it, endpos_it = tuples
                        for estimator in estimatorList:
                            estimator.updateEstimator((startpos_it, endpos_it), (startvec_it, endvec_it), (startEnergy_it, eHistory[it]), index_it, stepsize_it)
                        startpos_it = endpos_it
                        startvec_it = equi_vec
                        startEnergy_it = eHistory[it]

                else:  # No step was taken: don't do estimation
                    false_diff_counter += 1
                    diff_energy = kin_energy
                    diff_vec3d = kin_vec3d

                # set final state
                pos3d = diff_pos3d
                vec3d = diff_vec3d
                energy = diff_energy
                index = diff_index
            else:  # No energy was left
                pos3d = kin_pos3d
                vec3d = kin_vec3d
                energy = kin_energy
                index = kin_index

        self.updateAnalytics(kin_counter, diff_counter, false_diff_counter)

    def stepParticleDiffusive(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> tuple[tuple3d, Optional[tuple3d], int, bool, list[tuple[int, float, tuple3d]]]:
        """Apply diffusive motion to particle

        Args:
            pos3d (tuple3d): Position of particle after kinetic step
            vec3d (tuple3d): final velocity on which the diffusive moment is conditioned
            energy (float): energy of particle after kinetic step
            index (int): position of particle in estimator grid
            stepsize (float): remaing part of step for diffusive motion

        Returns:
            tuple[tuple3d, Optional[tuple3d], int, bool, list[tuple[int, float, tuple3d]]]:
                new_pos (tuple3d): New position after diffusive step
                equi_vec (Optional[tuple3d]): if relevant return the direction vector between the start and end point of the diffusive motion
                index (int): new index of particle after diffusive step
                diff_stepped (bool): if particle actually moved from previous position (eg. don't move if it ends up outside of the domain)
                diff_step_track (list[tuple[int, float, tuple3d]]): list containing tuples. Each tuple gives the index of a cell, the step the particle took in that cell and the endpoint of that step.
        """
        assert self.particle is not None

        # Get advection and diffusion coefficient
        A_coeff, D_coeff = self.advectionDiffusionCoeff(pos3d, vec3d, energy, index, stepsize)
        assert D_coeff[0] >= 0 and D_coeff[1] >= 0 and D_coeff[2] >= 0

        # Apply diffusive step
        xi = self.simOptions.rng.normal(size=(3, ))
        new_pos3d: tuple3d = pos3d + A_coeff*stepsize + np.sqrt(2*D_coeff*stepsize)*xi
        new_index = self.simDomain.getIndexPath(new_pos3d, vec3d)  # vector doesn't really matter here since particle won't be on an edge

        # Find equivalent kinetic step
        pos_delta = new_pos3d - pos3d
        equi_step = math.sqrt(pos_delta[0]**2 + pos_delta[1]**2 + pos_delta[2]**2)
        if equi_step == 0.0 or not self.simDomain.checkInDomain(new_pos3d):  # type: ignore
            return pos3d, None, index, False, [(index, 0.0, pos3d)]

        equi_vec = pos_delta/equi_step

        if index == new_index:
            return new_pos3d, equi_vec, index, True, [(index, stepsize, new_pos3d)]
        else:
            # For estimator: need stepsizes and indices of cells that are crossed.
            diff_step_track: list[tuple[int, float, tuple3d]] = []
            pos_it = pos3d
            index_it = index
            # Loop through cells between begin point and end point and write down the stepsizes
            while index_it != new_index:
                stepGeom, domainEdge, new_pos_geom = self.simDomain.getCellEdgeInformation(pos_it, equi_vec, index_it)  # diffusive step did not exceed boundaries
                if domainEdge == True:
                    return pos3d, None, index, False, [(index, 0.0, pos3d)]  # Switch to kinetic simulation to compute domain edge scattering event
                else:
                    diff_step_track.append((index_it, stepGeom, new_pos_geom))
                    pos_it = new_pos_geom
                    index_it = self.simDomain.getIndexPath(new_pos_geom, equi_vec)

            # record stepsize in final cell
            final_stepsize = math.sqrt((pos_it[0] - new_pos3d[0])**2 + (pos_it[1] - new_pos3d[1])**2 + (pos_it[2] - new_pos3d[2])**2)
            diff_step_track.append((index_it, final_stepsize, new_pos3d))
            return new_pos3d, equi_vec, new_index, True, diff_step_track

    @abstractmethod
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        """Return advection and diffusion coefficient

        Args:
            pos3d (tuple3d): position x double prime
            vec3d (tuple3d): final direction of travel on which the mean and variance are conditioned
            stepsize (float): remaining 'time' for the diffusive movement

        Returns:
            Tuple[tuple3d, tuple3d]: Advection and diffusion coefficient
        """
        pass


class KDMC(KDParticleTracer):
    """Implements kinetic-diffusion Monte Carlo using the mean and variance of kinetic motion conditioned on the final velocity.
    """
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        assert self.particle is not None

        # Get mean and variance of Omega distribution
        mu_omega, _ = self.particle.getOmegaMoments(pos3d)

        # Fix intermediate position
        x_int = pos3d + mu_omega*stepsize/2  # type: ignore

        material = self.simDomain.getMaterial(index)
        mu_omega, var_omega = self.particle.getOmegaMoments(x_int)
        Sigma_s = self.particle.getScatteringRate(x_int, energy, material)

        # intermediate results
        stepsizeSigma2 = stepsize*(Sigma_s**2)
        sigmaStepsize = Sigma_s*stepsize
        sigmaStepsizeDouble = 2*sigmaStepsize
        exp1: float = math.exp(-sigmaStepsize)
        exp2: float = math.exp(-sigmaStepsizeDouble)
        vec_mean_dev = vec3d - mu_omega
        vec_mean_dev2 = vec_mean_dev**2

        # Heterogeneity correction
        het1 = 0.5*vec_mean_dev2*(exp2 + sigmaStepsizeDouble*exp1 - 1.0)/stepsizeSigma2
        het2 = var_omega*(2*exp1 + sigmaStepsize + sigmaStepsize*exp1 - 2.0)/stepsizeSigma2
        het3 = var_omega*(sigmaStepsize*exp1 - 1.0 + exp1)/Sigma_s
        het4 = vec_mean_dev2*(sigmaStepsize*exp1 - exp1 + exp2)/Sigma_s
        het_correction = het1 - het2 - het3 + het4

        # Mean
        dRdx = self.particle.getDScatteringRate(x_int, vec3d, energy, material)
        mean: tuple3d = mu_omega + (1 - exp1)*vec_mean_dev/sigmaStepsize + het_correction*dRdx

        # Variance
        temp1 = 1.0 - exp1*sigmaStepsizeDouble - exp2  # Due to catastrophic cancellation, this thing can become negative. In this case, put it to zero.
        if temp1 < 0: temp1 = 0
        var_term1 = vec_mean_dev2*temp1/(2*stepsizeSigma2)
        temp2 = 2*exp1 + sigmaStepsize + sigmaStepsize*exp1 - 2.0  # Due to catastrophic cancellation, this thing can become negative. In this case, put it to zero.
        if temp2 < 0: temp2 = 0
        var_term2 = var_omega*temp2/stepsizeSigma2

        return mean, var_term1 + var_term2


class KDSMC(KDParticleTracer):
    """Implements KDMC using a mean and variance that doesn't incorporate the correlation between multiple time steps. The 's' stands for 'simple'.
    """
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        assert self.particle is not None

        # Get mean and variance of Omega distribution
        mu_omega, _ = self.particle.getOmegaMoments(pos3d)

        # Fix intermediate position
        x_int = pos3d + mu_omega*stepsize/2  # type: ignore

        material = self.simDomain.getMaterial(index)
        mu_omega, var_omega = self.particle.getOmegaMoments(x_int)
        Sigma_s = self.particle.getScatteringRate(x_int, energy, material)

        # Intermediate results
        sigmaStepsize = Sigma_s*stepsize
        sigma2Stepsize = (Sigma_s**2)*stepsize
        exp1 = math.exp(-sigmaStepsize)

        # Variance (divide by 2*stepsize)
        var = var_omega*(sigmaStepsize - 1.0 + exp1)/sigma2Stepsize

        # Mean
        dRdx = self.particle.getDScatteringRate(x_int, vec3d, energy, material)
        mean = mu_omega - dRdx*var_omega*(exp1 - 1.0 + sigmaStepsize*exp1)/sigma2Stepsize

        return mean, var


class KDLMC(KDParticleTracer):
    """Implements KDMC using the advection and diffusion coefficients from the diffusion limit.
    """
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        assert self.particle is not None

        # Get mean and variance of Omega distribution
        mu_omega, _ = self.particle.getOmegaMoments(pos3d)

        # Fix intermediate position
        x_int = pos3d + mu_omega*stepsize/2  # type: ignore

        material = self.simDomain.getMaterial(index)
        mu_omega, var_omega = self.particle.getOmegaMoments(x_int)
        assert np.all(mu_omega == 0), 'Diffusion limit is only valid in case mu_omega is zero.'

        Sigma_s = self.particle.getScatteringRate(x_int, energy, material)
        dRdx = self.particle.getDScatteringRate(x_int, vec3d, energy, material)

        mean = - var_omega*dRdx/(stepsize*(Sigma_s**2))
        var = var_omega/(Sigma_s*2*stepsize)

        return mean, var


class KDR(KDParticleTracer):
    """
    Version of kinetic diffusion Monte Carlo where the mean is conditioned on the velocity of the kinetic step. This is preferred for velocities that are rotations of the previous velocity. The variance is of the diffusion step is taken from the particle object. The particle object has the variance stored in a look-up table since no analytic formula for the variance of many kinetic steps exists.
    """
    def advectionDiffusionCoeff(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> Tuple[tuple3d, tuple3d]:
        assert self.particle is not None

        u, v, w = vec3d
        material = self.simDomain.getMaterial(index)

        # Load variance from LUT
        varmu, varsint = self.particle.getScatteringVariance(energy, stepsize, material)

        # Variance of isotropic scattering angle phi
        varcosphi = varsinphi = 0.5

        # Rotate variance to pre-diffusive direction
        varRotated = np.empty(shape=(3, ), dtype=float)
        if math.isclose(abs(vec3d[2]), 1.0, rel_tol=1e-14):  # indeterminate case
            sign = math.copysign(1.0, w)
            varRotated[0] = abs(sign*varsint*varcosphi)
            varRotated[1] = abs(sign*varsint*varsinphi)
            varRotated[2] = abs(sign*varmu)
        else:
            temp = math.sqrt(1 - w**2)
            varRotated[0] = abs(u*varmu + varsint*(u*w*varcosphi - v*varsinphi)/temp)
            varRotated[1] = abs(v*varmu + varsint*(v*w*varcosphi + u*varsinphi)/temp)
            varRotated[2] = abs(w*varmu - temp*varsint*varcosphi)

        return vec3d, varRotated/(stepsize*2)

    def stepParticleDiffusive(self, pos3d: tuple3d, vec3d: tuple3d, energy: float, index: int, stepsize: float) -> tuple[tuple3d, Optional[tuple3d], int, bool, list[tuple[int, float, tuple3d]]]:
        """Apply diffusive motion to particle

        Args:
            pos3d (tuple3d): Position of particle after kinetic step
            vec3d (tuple3d): final velocity on which the diffusive moment is conditioned
            energy (float): energy of particle after kinetic step
            index (int): position of particle in estimator grid
            stepsize (float): remaing part of step for diffusive motion

        Returns:
            tuple[tuple3d, Optional[tuple3d], int, bool, list[tuple[int, float, tuple3d]]]:
                new_pos (tuple3d): New position after diffusive step
                equi_vec (Optional[tuple3d]): if relevant return the direction vector between the start and end point of the diffusive motion
                index (int): new index of particle after diffusive step
                diff_stepped (bool): if particle actually moved from previous position (eg. don't move if it ends up outside of the domain)
                diff_step_track (list[tuple[int, float, tuple3d]]): list containing tuples. Each tuple gives the index of a cell, the step the particle took in that cell and the endpoint of that step.
        """
        assert self.particle is not None

        # Get advection and diffusion coefficient
        A_coeff, D_coeff = self.advectionDiffusionCoeff(pos3d, vec3d, energy, index, stepsize)
        assert D_coeff[0] >= 0 and D_coeff[1] >= 0 and D_coeff[2] >= 0

        # Apply diffusive step
        xi = self.simOptions.rng.normal(size=(3, ))
        new_pos3d: tuple3d = pos3d + A_coeff*stepsize + np.sqrt(2*D_coeff*stepsize)*xi
        new_index = self.simDomain.getIndexPath(new_pos3d, vec3d)  # vector doesn't really matter here since particle won't be on an edge

        # Find equivalent kinetic step
        pos_delta = new_pos3d - pos3d
        equi_step = math.sqrt(pos_delta[0]**2 + pos_delta[1]**2 + pos_delta[2]**2)
        if equi_step == 0.0 or not self.simDomain.checkInDomain(new_pos3d):  # type: ignore
            return pos3d, None, index, False, [(index, 0.0, pos3d)]

        equi_vec = pos_delta/equi_step

        if index == new_index:
            return new_pos3d, equi_vec, index, True, [(index, stepsize, new_pos3d)]
        else:
            # For estimator: need stepsizes and indices of cells that are crossed.
            diff_step_track: list[tuple[int, float, tuple3d]] = []
            pos_it = pos3d
            index_it = index
            # Loop through cells between begin point and end point and write down the stepsizes
            while index_it != new_index:
                stepGeom, domainEdge, new_pos_geom = self.simDomain.getCellEdgeInformation(pos_it, equi_vec, index_it)  # diffusive step did not exceed boundaries
                if domainEdge == True:
                    return pos3d, None, index, False, [(index, 0.0, pos3d)]  # Switch to kinetic simulation to compute domain edge scattering event
                else:
                    diff_step_track.append((index_it, stepGeom, new_pos_geom))
                    pos_it = new_pos_geom
                    index_it = self.simDomain.getIndexPath(new_pos_geom, equi_vec)

            # record stepsize in final cell
            final_stepsize = math.sqrt((pos_it[0] - new_pos3d[0])**2 + (pos_it[1] - new_pos3d[1])**2 + (pos_it[2] - new_pos3d[2])**2)
            diff_step_track.append((index_it, final_stepsize, new_pos3d))
            return new_pos3d, equi_vec, new_index, True, diff_step_track


    def traceParticle(self, estimators: Union[MCEstimator, tuple[MCEstimator, ...]]) -> None:
        """Step particle through simulation domain until its energy is below a threshold value. Estimator routine is called after each step for on-the-fly estimation.

        Args:
            estimator (MCEstimator): Estimator object which implements scoring of all quantities of interest

        """
        assert self.dS is not None
        assert self.particle is not None
        if isinstance(estimators, MCEstimator):  # estimator is now a list of estimators
            estimatorList: Tuple[MCEstimator, ...] = (estimators, )
        else:
            estimatorList = estimators

        # Sample initial condition in 2D
        pos3d: tuple3d = self.simOptions.initialPosition()
        vec3d: tuple3d = self.simOptions.initialDirection()
        energy: float = self.simOptions.initialEnergy()
        index: int = self.simDomain.getIndexPath(pos3d, vec3d)

        loopbool: bool = True

        # Couter kinetic and diffusive steps
        kin_counter: int = 0
        diff_counter: int = 0
        false_diff_counter: int = 0

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

            if loopbool:  # Do diffusive step if there is energy left

                step_diff1, diff_energy1 = self.pickStepSize(kin_pos3d, kin_energy, kin_index, step_kin)

                diff_pos3d, equi_vec, diff_index, diff_stepped, diff_step_track,  = self.stepParticleDiffusive(kin_pos3d, kin_vec3d, kin_energy, kin_index, step_diff1)
                if diff_stepped:  # Step was taken: do estimation
                    assert equi_vec is not None

                    diff_counter += 1
                    diff_energy = diff_energy1

                    # If energy left
                    if diff_energy <= self.simOptions.minEnergy: # have estimator deposit all remaining energy
                        if self.simOptions.DEPOSIT_REMAINDING_E_LOCALLY:
                            diff_energy = 0
                        loopbool = False  # make this the last iteration

                    # sample new direction for future kinetic step
                    mu, phi, new_direction_bool = self.particle.sampleScatteringAngles(diff_energy, self.simDomain.getMaterial(index))
                    diff_vec3d = self.scatterParticle(mu, phi, equi_vec, new_direction_bool)

                    # Divide energy loss evenly over all crossed cells based on stepsize
                    dE = kin_energy - diff_energy
                    steps = np.array([x[1] for x in diff_step_track], dtype=float)
                    eHistory = kin_energy - np.cumsum(steps*dE/steps.sum())  # type: ignore
                    eHistory[-1] = diff_energy

                    # Score QOIs: score each track in each cell that was crossed seperatly
                    startpos_it = kin_pos3d
                    startvec_it = kin_vec3d
                    endvec_it = equi_vec
                    startEnergy_it = kin_energy
                    for it, tuples in enumerate(diff_step_track):
                        if it == len(diff_step_track)-1:
                            endvec_it = diff_vec3d
                        index_it, stepsize_it, endpos_it = tuples
                        for estimator in estimatorList:
                            estimator.updateEstimator((startpos_it, endpos_it), (startvec_it, endvec_it), (startEnergy_it, eHistory[it]), index_it, stepsize_it)
                        startpos_it = endpos_it
                        startvec_it = equi_vec
                        startEnergy_it = eHistory[it]

                else:  # No step was taken: don't do estimation
                    false_diff_counter += 1
                    diff_energy = kin_energy
                    diff_vec3d = kin_vec3d

                # set final state
                pos3d = diff_pos3d
                vec3d = diff_vec3d
                energy = diff_energy
                index = diff_index
            else:  # No energy was left
                pos3d = kin_pos3d
                vec3d = kin_vec3d
                energy = kin_energy
                index = kin_index

        self.updateAnalytics(kin_counter, diff_counter, false_diff_counter)
