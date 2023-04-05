import sys
import pickle
import time
import numpy as np
from mpi4py import MPI
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.SimOptions import KDTestSource
from electronTransportCode.ParticleModel import DiffusionTestParticle
from electronTransportCode.MCParticleTracer import AnalogParticleTracer, KDMC, KDSMC
from electronTransportCode.MCEstimator import TrackEndEstimator


# Set up simulation domain
xmax = 15
ymin = -xmax; ymax = xmax; ybins = 1  # No internal grid cell crossings
zmin = -xmax; zmax = xmax; zbins = 1
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

# Set up initial conditions
SEED: int = 4  # Random number generator seed

scatteringRate1 = 1.0
particle1 = DiffusionTestParticle(Es=scatteringRate1, sp=1.0)

if __name__ == '__main__':
    myrank = MPI.COMM_WORLD.Get_rank()

    nbSims = 10
    dsArray = np.logspace(-2, 0, nbSims)

    nproc = MPI.COMM_WORLD.Get_size()
    eSource = 5.0

    pointSourceSim = KDTestSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource)
    particleTracerK = AnalogParticleTracer(particle=None, simOptions=pointSourceSim, simDomain=simDomain)
    particleTracerKD = KDMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS = None)
    particleTracerKDS = KDSMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS = None)

    NB_PARTICLES = int(sys.argv[1])
    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/nproc)
    NB_PARTICLES = NB_PARTICLES_PER_PROC*nproc

    TrackEndEstimatorK1 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x')
    TEKDList = [TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x') for _ in range(nbSims)]
    TEKDSList = [TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x') for _ in range(nbSims)]

    t1 = time.perf_counter()

    # Run KDS particle tracer
    for i in range(nbSims):
        particleTracerKDS.dS = dsArray[i]
        t3 = time.perf_counter()
        particleTracerKDS.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TEKDSList[i], ), particle=particle1, file=f'data/TrackEndEstimatorKDS{i}.pkl', verbose=False)
        t4 = t4 = time.perf_counter()
        if myrank == 0: print(f'KDSMC ds: {dsArray[i]} took {round(t4-t3, 4)}s')

    # Run analog particle tracer
    t3 = time.perf_counter()
    particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorK1, ), particle=particle1, file='data/TrackEndEstimatorK.pkl', verbose=False)
    t4 = time.perf_counter()
    if myrank == 0: print(f'Analog particle tracer took {round(t4-t3, 4)}s')

    # Run KD particle tracer
    for i in range(nbSims):
        particleTracerKD.dS = dsArray[i]
        t3 = time.perf_counter()
        particleTracerKD.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TEKDList[i], ), particle=particle1, file=f'data/TrackEndEstimatorKD{i}.pkl', verbose=False)
        t4 = time.perf_counter()
        if myrank == 0: print(f'KDMC ds: {dsArray[i]} took {round(t4-t3, 4)}s')

    t2 = time.perf_counter()

    if myrank == 0:
        print('\n')
        print(f'Simulation took {round(t2-t1, 4)} seconds. Writing results...')

        # dump argv
        tup = (eSource, NB_PARTICLES, dsArray)
        pickle.dump(tup, open('data/simargv.pkl', 'wb'))

        # dump particle tracers
        pickle.dump(particleTracerK, open('data/particleTracerK.pkl', 'wb'))
        pickle.dump(particleTracerKD, open('data/particleTracerKD.pkl', 'wb'))
        pickle.dump(particleTracerKDS, open('data/particleTracerKDS.pkl', 'wb'))
