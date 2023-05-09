import sys
import pickle
import time
from mpi4py import MPI
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.SimOptions import PointSource
from electronTransportCode.ParticleModel import DiffusionTestParticle, DiffusionTestParticlev2
from electronTransportCode.MCParticleTracer import AnalogParticleTracer, KDMC
from electronTransportCode.MCEstimator import MomentumTypeEstimator


# Set up simulation domain
xmax = 10
ymin = -xmax; ymax = xmax; ybins = 200  # No internal grid cell crossings
zmin = -xmax; zmax = xmax; zbins = 200
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

# Set up initial conditions
SEED: int = 4  # Random number generator seed

scatteringRate = 10.0

if __name__ == '__main__':

    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()
    eSource = float(sys.argv[2])
    particleType = int(sys.argv[3])

    if particleType == 1:
        particle = DiffusionTestParticle(Es=scatteringRate, sp=1.0)
        fileK = 'data/TrackEndEstimatorK1.pkl'
        fileKD = 'data/TrackEndEstimatorKD1.pkl'
        simargvFile = 'data/simargv1.pkl'
    elif particleType == 2:
        particle = DiffusionTestParticlev2(Es=scatteringRate, sp=1.0)
        fileK = 'data/TrackEndEstimatorK2.pkl'
        fileKD = 'data/TrackEndEstimatorKD2.pkl'
        simargvFile = 'data/simargv2.pkl'
    else:
        raise ValueError('Invalid particle type')

    pointSourceSim = PointSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource)
    particleTracerK = AnalogParticleTracer(particle=particle, simOptions=pointSourceSim, simDomain=simDomain)
    particleTracerKD = KDMC(particle=particle, simOptions=pointSourceSim, simDomain=simDomain, dS = 0.2)  # stepsize is final time!

    NB_PARTICLES = int(float(sys.argv[1]))
    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/nproc)
    NB_PARTICLES = NB_PARTICLES_PER_PROC*nproc

    momEstimatorK1 = MomentumTypeEstimator(simDomain=simDomain)
    momEstimatorKD1 = MomentumTypeEstimator(simDomain=simDomain)
    momEstimatorK2 = MomentumTypeEstimator(simDomain=simDomain)
    momEstimatorKD2 = MomentumTypeEstimator(simDomain=simDomain)

    t1 = time.perf_counter()

    # Run analog particle tracer
    logAmount = int(NB_PARTICLES_PER_PROC/10)
    particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=(momEstimatorK1, ), file=fileK, logAmount=logAmount)

    # Run KD particle tracer
    particleTracerKD.runMultiProc(nbParticles=NB_PARTICLES, estimators=(momEstimatorKD1, ), file=fileKD, logAmount=logAmount)

    t2 = time.perf_counter()

    if rank == 0:
        print('\n')
        print(f'Simulation took {t2-t1} seconds. Writing results...')

        # dump argv
        tup = (eSource, NB_PARTICLES)
        pickle.dump(tup, open(simargvFile, 'wb'))

        # dump particle tracers
        pickle.dump(particleTracerK, open('data/particleTracerK.pkl', 'wb'))
        pickle.dump(particleTracerKD, open('data/particleTracerKD.pkl', 'wb'))

    # run: mpiexec -n 4 python3 -m simpleHomogeneous 1.0 30000
