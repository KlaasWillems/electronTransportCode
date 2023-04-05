import sys
import pickle
import time
import numpy as np
from mpi4py import MPI
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.SimOptions import KDTestSource
from electronTransportCode.ParticleModel import DiffusionTestParticlev2
from electronTransportCode.MCParticleTracer import AnalogParticleTracer, KDMC
from electronTransportCode.MCEstimator import TrackEndEstimator


# Set up simulation domain
xmax = 50000
ymin = -xmax; ymax = xmax; ybins = 1  # No internal grid cell crossings
zmin = -xmax; zmax = xmax; zbins = 1
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

# Set up initial conditions
SEED: int = 4  # Random number generator seed
sp = 1.0


if __name__ == '__main__':

    # --- Set up variables and objects
    nproc = MPI.COMM_WORLD.Get_size()
    eSource = 5.0
    myrank = MPI.COMM_WORLD.Get_rank()

    pointSourceSim = KDTestSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource)
    particleTracerK = AnalogParticleTracer(particle=None, simOptions=pointSourceSim, simDomain=simDomain)
    particleTracerKD = KDMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS = eSource)  # stepsize is final time!

    NB_PARTICLES = 50000
    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/nproc)

    # --- Set up estimators
    nbSim = 10
    scatteringRateList = np.logspace(-2, 2, nbSim)
    particleList = [DiffusionTestParticlev2(Es=scatteringRateList[i], sp=sp) for i in range(nbSim)]
    warmUpEstimator = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x')

    # --- Timing results
    repeats = 1
    timingsK = np.zeros((nbSim, repeats))
    timingsKD = np.zeros((nbSim, repeats))

    # --- Collision results
    collisionsK = np.zeros((nbSim, ))  # Average amount of collisions for a particle
    collisionsKD = np.zeros((nbSim, 2))  # Average amount of kinetic and diffusive collisions of a particle

    # --- Warm-up run
    if myrank == 0:
        print('Doing warm-up simulation...')
    particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=warmUpEstimator, particle=particleList[int(nbSim/2)], file=None, verbose=False)

    # --- Run simulations
    t1 = time.perf_counter()

    for sim in range(nbSim):
        for repeat in range(repeats):
            trackEndEstimatorKList = [TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x') for i in range(nbSim)]
            trackEndEstimatorKDList = [TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x') for i in range(nbSim)]

            MPI.COMM_WORLD.Barrier()
            t3 = time.perf_counter()

            # Run analog particle tracer
            particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=trackEndEstimatorKList[sim], particle=particleList[sim], file=None, verbose=False)

            MPI.COMM_WORLD.Barrier()
            t4 = time.perf_counter()

            # Run KD particle tracer
            particleTracerKD.runMultiProc(nbParticles=NB_PARTICLES, estimators=trackEndEstimatorKDList[sim], particle=particleList[sim], file=None, verbose=False)

            MPI.COMM_WORLD.Barrier()
            t5 = time.perf_counter()

            # store timings. Only root process will eventually write results to file.
            timingsK[sim, repeat] = t4-t3
            timingsKD[sim, repeat] = t5-t4
            if myrank == 0:
                print(f'Scattering rate: {round(scatteringRateList[sim], 4)}, repeat: {repeat}, K time: {round(timingsK[sim, repeat], 4)}s, KD time: {round(timingsKD[sim, repeat], 4)}s')

            # Store collision data
            if repeat == repeats-1:
                collisionsK[sim] = particleTracerK.averageNbCollisions
                collisionsKD[sim, 0] = particleTracerKD.AvgNbAnalogCollisions
                collisionsKD[sim, 1] = particleTracerKD.AvgNbDiffCollisions

    t2 = time.perf_counter()

    # --- Write timings to file
    if myrank == 0:
        print('\n')
        print(f'Test took {t2-t1} seconds. Writing results...')

        # dump argv
        tup = (timingsK, timingsKD, scatteringRateList, particleTracerKD.dS, collisionsK, collisionsKD)
        pickle.dump(tup, open('data/timings.pkl', 'wb'))
