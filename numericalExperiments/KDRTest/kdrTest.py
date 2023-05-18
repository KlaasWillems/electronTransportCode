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
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron, KDRTestParticle
from electronTransportCode.SimOptions import PointSource
from electronTransportCode.MCParticleTracer import KDR, AnalogParticleTracer

# Command line arguments
# 1) Amount of particles to simulate
# 2) Simulation type 'k' or 'kdr'
# 3) Amount of steps \Delta s to simulate a particle
# 4) If simulation type is 'kdr', final argument is a boolean. True if multiple scattering angle should be used. False if not.


# Initialize simulation parameters
xmax = 15

# Set up simulation domain
ymin = -xmax; ymax = xmax; ybins = 1
zmin = -xmax; zmax = xmax; zbins = 1
material = Material(rho=1.05)
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=material)
stepsize = 0.1

if __name__ == '__main__':
    myrank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()

    NB_PARTICLES_PER_PROC = int(float(sys.argv[1])/nproc)
    NB_PARTICLES = int(NB_PARTICLES_PER_PROC*nproc)

    simType = sys.argv[2]
    factor = int(float(sys.argv[3]))

    # Set up simDomain
    particle1 = KDRTestParticle()
    deltaE = particle1.energyLoss(1.0, None, stepsize*factor, material)  # type: ignore

    # Set up initial conditions
    eSource: float = deltaE
    SEED: int = 4  # Random number generator seed
    pointSourceSim = PointSource(minEnergy=0.0, rngSeed=myrank, eSource=eSource)

    logAmount = int(NB_PARTICLES_PER_PROC/10)
    # - Run simulation
    if simType == 'k':
        trackEndEstimatorkx = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
        trackEndEstimatorky = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='y')
        trackEndEstimatorkz = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='z')
        kineticParticleTracer = AnalogParticleTracer(particle=particle1, simOptions=pointSourceSim, simDomain=simDomain)
        t1 = time.process_time()
        kineticParticleTracer.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorkx, trackEndEstimatorky, trackEndEstimatorkz), file=f'data/trackEndEstimatork{factor}.pkl', logAmount=logAmount)
        t2 = time.process_time()
        print(f'Kinetic simulation time: {round(t2-t1, 4)}')
        if MPI.COMM_WORLD.Get_rank() == 0:
            pickle.dump(kineticParticleTracer, open(f'data/particleTracerK{factor}.pkl', 'wb'))
    elif simType == 'kdr':
        trackEndEstimatorkdrx = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
        trackEndEstimatorkdry = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='y')
        trackEndEstimatorkdrz = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='z')
        MS = bool(sys.argv[4])
        if MS:
            file = f'data/trackEndEstimatorkdrMS{factor}.pkl'
        else:
            file = f'data/trackEndEstimatorkdr{factor}.pkl'
        kdr = KDR(simOptions=pointSourceSim, simDomain=simDomain, particle=particle1, dS=stepsize, useMSAngle=MS)
        t2 = time.process_time()
        kdr.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorkdrx, trackEndEstimatorkdry, trackEndEstimatorkdrz), file=file, logAmount=logAmount)
        t3 = time.process_time()
        print(f'KDR simulation time: {round(t3-t2, 4)}')
        if MPI.COMM_WORLD.Get_rank() == 0:
            pickle.dump(kdr, open(f'data/kdr{factor}.pkl', 'wb'))

    # dump argv
    tup = (eSource, NB_PARTICLES)
    pickle.dump(tup, open('data/simargv.pkl', 'wb'))
