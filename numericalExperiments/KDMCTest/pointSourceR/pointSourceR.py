import sys
import pickle
import time
import numpy as np
from mpi4py import MPI
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.SimOptions import PointSource
from electronTransportCode.ParticleModel import PointSourceParticle
from electronTransportCode.MCParticleTracer import AnalogParticleTracer, KDMC, KDSMC
from electronTransportCode.MCEstimator import TrackEndEstimator


# Set up simulation domain
xmax = 20
ymin = -xmax; ymax = xmax; ybins = 1  # No internal grid cell crossings
zmin = -xmax; zmax = xmax; zbins = 1
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

# Set up initial conditions
SEED: int = 4  # Random number generator seed


if __name__ == '__main__':
    myrank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()

    eSource = 5.0
    nbdS = 10
    repeats = int(sys.argv[2])
    dS = 1.0
    sigmaArray = np.logspace(-2, 2, 10)
    particleList = [PointSourceParticle(generator=SEED, sigma=sigma) for sigma in sigmaArray]

    pointSourceSim = PointSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource)
    particleTracerK = AnalogParticleTracer(particle=None, simOptions=pointSourceSim, simDomain=simDomain)
    particleTracerKD = KDMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS=dS)
    particleTracerKDS = KDSMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS=dS)

    NB_PARTICLES = int(sys.argv[1])
    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/nproc)
    NB_PARTICLES = NB_PARTICLES_PER_PROC*nproc

    t1 = time.perf_counter()

    # Run analog particle tracer
    for i, sigma in enumerate(sigmaArray):
        for repeat in range(repeats):
            TrackEndEstimatorK = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x')
            t3 = time.perf_counter()
            particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorK, ), particle=particleList[i], file=f'data/TEEAnalog{i}_{repeat}.pkl', verbose=False)
            t4 = time.perf_counter()
            if myrank == 0: print(f'Analog: sigma: {sigma:.2e}, repeat: {repeat}, simulation time: {round(t4-t3, 4)}s')

    # Run KD particle tracer
    for i, sigma in enumerate(sigmaArray):
        for repeat in range(repeats):
            TrackEndEstimatorK = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x')
            t3 = time.perf_counter()
            particleTracerKD.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorK, ), particle=particleList[i], file=f'data/TEEKDMC{i}_{repeat}.pkl', verbose=False)
            t4 = time.perf_counter()
            if myrank == 0: print(f'KDMC: sigma: {sigma:.2e}, repeat: {repeat}, simulation time: {round(t4-t3, 4)}s')

    # Run KD particle tracer
    for i, sigma in enumerate(sigmaArray):
        for repeat in range(repeats):
            TrackEndEstimatorK = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='x')
            t3 = time.perf_counter()
            particleTracerKDS.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorK, ), particle=particleList[i], file=f'data/TEEKDSMC{i}_{repeat}.pkl', verbose=False)
            t4 = time.perf_counter()
            if myrank == 0: print(f'KDMC: sigma: {sigma:.2e}, repeat: {repeat}, simulation time: {round(t4-t3, 4)}s')

    t2 = time.perf_counter()

    if myrank == 0:
        print('\n')
        print(f'Simulation took {round(t2-t1, 4)} seconds. Writing results...')

        # dump argv
        tup = (eSource, dS, NB_PARTICLES, sigmaArray, repeats)
        pickle.dump(tup, open('data/simargv.pkl', 'wb'))

        # dump particle tracers
        pickle.dump(particleTracerK, open('data/particleTracerK.pkl', 'wb'))
        pickle.dump(particleTracerKD, open('data/particleTracerKD.pkl', 'wb'))
        pickle.dump(particleTracerKDS, open('data/particleTracerKDS.pkl', 'wb'))
