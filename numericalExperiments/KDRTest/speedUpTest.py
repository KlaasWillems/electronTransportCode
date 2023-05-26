import sys
import pickle
import time
import numpy as np
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import Material
from electronTransportCode.SimOptions import KDTestSource
from electronTransportCode.ParticleModel import KDRSpeedUpParticle
from electronTransportCode.MCParticleTracer import KDR, AnalogParticleTracer
from electronTransportCode.MCEstimator import TrackEndEstimator


# Initialize simulation parameters
xmax = 15

# Set up simulation domain
ymin = -xmax; ymax = xmax; ybins = 1
zmin = -xmax; zmax = xmax; zbins = 1
material = Material(rho=1.05)
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=material)


if __name__ == '__main__':

    # --- Set up variables and objects
    eSource = 5.0
    SEED = 2
    dS = 0.1
    MSDistribution: str = 'lognormal'

    pointSourceSim = KDTestSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource)
    particleTracerK = AnalogParticleTracer(particle=None, simOptions=pointSourceSim, simDomain=simDomain)
    kdr = KDR(simOptions=pointSourceSim, simDomain=simDomain, dS=dS, useMSAngle=True, particle=None)

    NB_PARTICLES = int(sys.argv[1])

    # --- Set up estimators
    nbSim = 20
    scatteringRateList = np.logspace(3.5, -3.5, nbSim)
    particleList = [KDRSpeedUpParticle(sigma=scatteringRateList[i], msDist=MSDistribution) for i in range(nbSim)]
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
            kdr.__call__(nbParticles=NB_PARTICLES, estimators=trackEndEstimatorKDList[sim], particle=particleList[sim])

            t5 = time.perf_counter()

            # store timings. Only root process will eventually write results to file.
            timingsK[sim, repeat] = t4-t3
            timingsKD[sim, repeat] = t5-t4
            print(f'Scattering rate: {round(scatteringRateList[sim], 4)}, repeat: {repeat}, K time: {round(timingsK[sim, repeat], 4)}s, KD time: {round(timingsKD[sim, repeat], 4)}s')

            # Store collision data
            if repeat == repeats-1:
                collisionsK[sim] = particleTracerK.averageNbCollisions
                collisionsKD[sim, 0] = kdr.AvgNbAnalogCollisions
                collisionsKD[sim, 1] = kdr.AvgNbDiffCollisions

    t2 = time.perf_counter()

    # --- Write timings to file
    print('\n')
    print(f'Test took {t2-t1} seconds. Writing results...')

    # dump argv
    tup = (timingsK, timingsKD, scatteringRateList, kdr.dS, collisionsK, collisionsKD)
    pickle.dump(tup, open('data/speedUp/timings.pkl', 'wb'))
