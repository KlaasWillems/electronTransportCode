import sys
import pickle
import time
import numpy as np
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


if __name__ == '__main__':

    # --- Set up variables and objects
    eSource = 5.0
    SEED = 2

    pointSourceSim = KDTestSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource)
    particleTracerK = AnalogParticleTracer(particle=None, simOptions=pointSourceSim, simDomain=simDomain)
    particleTracerKD = KDMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS = eSource)  # stepsize is final time!

    NB_PARTICLES = int(sys.argv[1])

    # --- Set up estimators
    sp = 1.0
    nbSim = 10
    scatteringRateList = np.logspace(2, -2, nbSim)
    particleList = [DiffusionTestParticlev2(Es=scatteringRateList[i], sp=sp) for i in range(nbSim)]
    warmUpEstimator = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES, setting='x')

    # --- Timing results
    repeats = 5
    timingsK = np.zeros((nbSim, repeats))
    timingsKD = np.zeros((nbSim, repeats))

    # --- Collision results
    collisionsK = np.zeros((nbSim, ))  # Average amount of collisions for a particle
    collisionsKD = np.zeros((nbSim, 2))  # Average amount of kinetic and diffusive collisions of a particle

    # --- Warm-up run
    print('Doing warm-up simulation...')
    particleTracerK.__call__(nbParticles=NB_PARTICLES, estimators=warmUpEstimator, particle=particleList[int(nbSim/2)])

    # --- Run simulations
    t1 = time.perf_counter()

    for sim in range(nbSim):
        for repeat in range(repeats):
            trackEndEstimatorKList = [TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES, setting='x') for i in range(nbSim)]
            trackEndEstimatorKDList = [TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES, setting='x') for i in range(nbSim)]

            t3 = time.perf_counter()

            # Run analog particle tracer
            particleTracerK.__call__(nbParticles=NB_PARTICLES, estimators=trackEndEstimatorKList[sim], particle=particleList[sim])

            t4 = time.perf_counter()

            # Run KD particle tracer
            particleTracerKD.__call__(nbParticles=NB_PARTICLES, estimators=trackEndEstimatorKDList[sim], particle=particleList[sim])

            t5 = time.perf_counter()

            # store timings. Only root process will eventually write results to file.
            timingsK[sim, repeat] = t4-t3
            timingsKD[sim, repeat] = t5-t4
            print(f'Scattering rate: {round(scatteringRateList[sim], 4)}, repeat: {repeat}, K time: {round(timingsK[sim, repeat], 4)}s, KD time: {round(timingsKD[sim, repeat], 4)}s')

            # Store collision data
            if repeat == repeats-1:
                collisionsK[sim] = particleTracerK.averageNbCollisions
                collisionsKD[sim, 0] = particleTracerKD.AvgNbAnalogCollisions
                collisionsKD[sim, 1] = particleTracerKD.AvgNbDiffCollisions

    t2 = time.perf_counter()

    # --- Write timings to file
    print('\n')
    print(f'Test took {t2-t1} seconds. Writing results...')

    # dump argv
    tup = (timingsK, timingsKD, scatteringRateList, particleTracerKD.dS, collisionsK, collisionsKD)
    pickle.dump(tup, open('data/timings.pkl', 'wb'))
