# Imports
import sys
import pickle
import numpy as np
from mpi4py import MPI
from pde import PDE, CartesianGrid, ScalarField
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.MCEstimator import TrackEndEstimator
from electronTransportCode.ParticleModel import DiffusionTestParticle
from electronTransportCode.SimOptions import DiffusionPointSource
from electronTransportCode.MCParticleTracer import AnalogParticleTracer

b = 0.2  # Parameters of transformation

# Initialize Advection-Diffusion parameters (No advection)
stoppingPower = f'{b}*(1 + E**2)'
scatteringRate1 = 1
scatteringRate2 = 10
scatteringRate3 = 50
varH = 1/3  # 1 in 1D, 1/3 in 3D
sigmaInit = 1
muInit = 0.0

# Initialize simulation parameters
xbins = 512
xmax = 15.0

# Initialize Monte Carlo Algorithm

# Set up simulation domain
ymin = -xmax; ymax = xmax; ybins = 2
zmin = -xmax; zmax = xmax; zbins = 2
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

# Set up initial conditions
SEED: int = 4  # Random number generator seed
Emax: float = 1.0  # Dummy
pointSourceSim = DiffusionPointSource(minEnergy=0.0, rngSeed=SEED, eSource=Emax, loc=muInit, std=sigmaInit)

particle1 = DiffusionTestParticle(Es=scatteringRate1, sp=stoppingPower)
particle2 = DiffusionTestParticle(Es=scatteringRate2, sp=stoppingPower)
particle3 = DiffusionTestParticle(Es=scatteringRate3, sp=stoppingPower)

particleTracer1 = AnalogParticleTracer(particle=particle1, simOptions=pointSourceSim, simDomain=simDomain)
particleTracer2 = AnalogParticleTracer(particle=particle2, simOptions=pointSourceSim, simDomain=simDomain)
particleTracer3 = AnalogParticleTracer(particle=particle3, simOptions=pointSourceSim, simDomain=simDomain)

file1 = 'trackEndEstimatorx1.pkl'
file2 = 'trackEndEstimatorx2.pkl'
file3 = 'trackEndEstimatorx3.pkl'

if __name__ == '__main__':
    # Get argv
    nproc = MPI.COMM_WORLD.Get_size()
    eSource1 = float(sys.argv[1])
    eSource2 = float(sys.argv[2])
    eSource3 = float(sys.argv[3])

    # set argv
    NB_PARTICLES_PER_PROC = int(int(sys.argv[4])/nproc)
    NB_PARTICLES = NB_PARTICLES_PER_PROC*nproc
    pointSourceSim.eSource = eSource1

    # Set up estimator and particle
    trackEndEstimatorx1 = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
    trackEndEstimatorx2 = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
    trackEndEstimatorx3 = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')

    # - Run simulation
    particleTracer1.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorx1, ), file=file1)
    pointSourceSim.eSource = eSource2
    particleTracer2.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorx2, ), file=file2)
    pointSourceSim.eSource = eSource3
    particleTracer3.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorx3, ), file=file3)

    # dump argv
    tup = (eSource1, eSource2, eSource3, NB_PARTICLES)
    pickle.dump(tup, open('simargv.pkl', 'wb'))
