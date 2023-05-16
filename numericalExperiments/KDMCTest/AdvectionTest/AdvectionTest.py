import sys
import pickle
import time
from mpi4py import MPI
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.SimOptions import KDTestSource
from electronTransportCode.ParticleModel import DiffusionTestParticlev2
from electronTransportCode.MCParticleTracer import AnalogParticleTracer, KDMC
from electronTransportCode.MCEstimator import TrackEndEstimator


# Set up simulation domain
xmax = 15
ymin = -xmax; ymax = xmax; ybins = 1  # No internal grid cell crossings
zmin = -xmax; zmax = xmax; zbins = 1
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

# Set up initial conditions
scatteringRate1 = '0.1*(1 + 0.5*sin(y))'; scatteringRate2 = '(1 + 0.5*sin(y))'; scatteringRate3 = '10*(1 + 0.5*sin(y))'
sp = 1.0
particle1 = DiffusionTestParticlev2(Es=scatteringRate1, sp=sp)
particle2 = DiffusionTestParticlev2(Es=scatteringRate2, sp=sp)
particle3 = DiffusionTestParticlev2(Es=scatteringRate3, sp=sp)

if __name__ == '__main__':

    nproc = MPI.COMM_WORLD.Get_size()
    myrank = MPI.COMM_WORLD.Get_rank()
    eSource = float(sys.argv[1])

    SEED = myrank

    pointSourceSim = KDTestSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource, dir='y')
    particleTracerK = AnalogParticleTracer(particle=None, simOptions=pointSourceSim, simDomain=simDomain)
    particleTracerKD1 = KDMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS = eSource, het_corr=True)
    particleTracerKD2 = KDMC(particle=None, simOptions=pointSourceSim, simDomain=simDomain, dS = eSource, het_corr=False)

    NB_PARTICLES = int(sys.argv[2])
    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/nproc)
    NB_PARTICLES = NB_PARTICLES_PER_PROC*nproc

    TrackEndEstimatorK1 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')
    TrackEndEstimatorK2 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')
    TrackEndEstimatorK3 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')

    TrackEndEstimatorKD1 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')
    TrackEndEstimatorKD2 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')
    TrackEndEstimatorKD3 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')
    TrackEndEstimatorKD4 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')
    TrackEndEstimatorKD5 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')
    TrackEndEstimatorKD6 = TrackEndEstimator(simDomain, nb_particles=NB_PARTICLES_PER_PROC, setting='y')

    t1 = time.perf_counter()

    # Run analog particle tracer
    logAmount = int(NB_PARTICLES_PER_PROC/10)
    particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorK1, ), particle=particle1, file='data/TrackEndEstimatorK1.pkl', logAmount=logAmount)
    particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorK2, ), particle=particle2, file='data/TrackEndEstimatorK2.pkl', logAmount=logAmount)
    particleTracerK.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorK3, ), particle=particle3, file='data/TrackEndEstimatorK3.pkl', logAmount=logAmount)

    # Run KD particle tracer
    particleTracerKD1.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorKD1, ), particle=particle1, file='data/TrackEndEstimatorKD1.pkl', logAmount=logAmount)
    particleTracerKD1.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorKD2, ), particle=particle2, file='data/TrackEndEstimatorKD2.pkl', logAmount=logAmount)
    particleTracerKD1.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorKD3, ), particle=particle3, file='data/TrackEndEstimatorKD3.pkl', logAmount=logAmount)
    particleTracerKD2.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorKD4, ), particle=particle1, file='data/TrackEndEstimatorKD4.pkl', logAmount=logAmount)
    particleTracerKD2.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorKD5, ), particle=particle2, file='data/TrackEndEstimatorKD5.pkl', logAmount=logAmount)
    particleTracerKD2.runMultiProc(nbParticles=NB_PARTICLES, estimators=(TrackEndEstimatorKD6, ), particle=particle3, file='data/TrackEndEstimatorKD6.pkl', logAmount=logAmount)

    t2 = time.perf_counter()

    if myrank == 0:

        print('\n')
        print(f'Simulation took {t2-t1} seconds. Writing results...')

        # dump argv
        tup = (eSource, NB_PARTICLES)
        pickle.dump(tup, open('data/simargv.pkl', 'wb'))

        # dump particle tracers
        pickle.dump(particleTracerK, open('data/particleTracerK.pkl', 'wb'))
        pickle.dump(particleTracerKD1, open('data/particleTracerKD1.pkl', 'wb'))
        pickle.dump(particleTracerKD2, open('data/particleTracerKD2.pkl', 'wb'))
