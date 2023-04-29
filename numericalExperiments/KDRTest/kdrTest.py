# Imports
import sys
import numpy as np
import pickle
import time
from mpi4py import MPI
from electronTransportCode.ProjectUtils import E_THRESHOLD
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import Material
from electronTransportCode.MCEstimator import TrackEndEstimator
from electronTransportCode.ParticleModel import KDRTestParticle
from electronTransportCode.SimOptions import PointSource
from electronTransportCode.MCParticleTracer import KDR, AnalogParticleTracer

# Initialize Advection-Diffusion parameters (No advection)
stoppingPower = 1
scatteringRate = 100
RNGSEED = 4

# Initialize simulation parameters
xmax = 15

# Set up simulation domain
ymin = -xmax; ymax = xmax; ybins = 1
zmin = -xmax; zmax = xmax; zbins = 1
material = Material(rho=1.05)
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=material)

# Set up initial conditions
eSource: float = 5.0
SEED: int = 4  # Random number generator seed
pointSourceSim = PointSource(minEnergy=E_THRESHOLD, rngSeed=RNGSEED, eSource=eSource)

particle1 = KDRTestParticle()

kineticParticleTracer = AnalogParticleTracer(particle=particle1, simOptions=pointSourceSim, simDomain=simDomain)
kdr = KDR(simOptions=pointSourceSim, simDomain=simDomain, particle=particle1, dS=0.1)

if __name__ == '__main__':
    nproc = MPI.COMM_WORLD.Get_size()

    NB_PARTICLES_PER_PROC = int(float(sys.argv[1])/nproc)
    NB_PARTICLES = int(NB_PARTICLES_PER_PROC*nproc)

    simType = sys.argv[2]

    # - Set up estimator and particle
    trackEndEstimatorkx = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
    trackEndEstimatorky = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='y')
    trackEndEstimatorkz = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='z')
    trackEndEstimatorkdrx = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
    trackEndEstimatorkdry = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='y')
    trackEndEstimatorkdrz = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='z')

    logAmount = int(NB_PARTICLES_PER_PROC/10)
    # - Run simulation
    if simType == 'k':
        t1 = time.process_time()
        kineticParticleTracer.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorkx, trackEndEstimatorky, trackEndEstimatorkz), file='data/trackEndEstimatork.pkl', logAmount=logAmount)
        t2 = time.process_time()
        print(f'Kinetic simulation time: {round(t2-t1, 4)}')
        if MPI.COMM_WORLD.Get_rank() == 0:
            pickle.dump(kineticParticleTracer, open('data/particleTracerK.pkl', 'wb'))
    elif simType == 'kdr':
        t2 = time.process_time()
        kdr.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorkdrx, trackEndEstimatorkdry, trackEndEstimatorkdrz), file='data/trackEndEstimatorkdr.pkl', logAmount=logAmount)
        t3 = time.process_time()
        print(f'KDR simulation time: {round(t3-t2, 4)}')
        if MPI.COMM_WORLD.Get_rank() == 0:
            pickle.dump(kdr, open('data/kdr.pkl', 'wb'))

    # dump argv
    tup = (eSource, NB_PARTICLES)
    pickle.dump(tup, open('data/simargv.pkl', 'wb'))