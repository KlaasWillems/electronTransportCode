# Imports
import sys
import os
import numpy as np
from mpi4py import MPI
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.abspath('..'),os.pardir))
sys.path.append(PROJECT_ROOT)

from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.MCEstimator import FluenceEstimator, DoseEstimator, TrackEndEstimator
from electronTransportCode.ParticleModel import DiffusionTestParticle
from electronTransportCode.SimOptions import DiffusionPointSource
from electronTransportCode.MCParticleTracer import AnalogParticleTracer

# Initialize Advection-Diffusion parameters
stoppingPower = 1
scatteringRate1 = '(1 + 0.5*sin(x))'; scatteringRate1_dx = '(-0.5*cos(x)/((1 + 0.5*sin(x))**2))'
scatteringRate2 = '(100 + 10*sin(x))'; scatteringRate2_dx = '(cos(x)/(10*(10 + sin(x))**2))'
varH = 1/3  # 1 in 1D, 1/3 in 3D
sigmaInit = 1.0
muInit = 0.0

# Initialize simulation parameters
tmax = 1.0
xbins = 512
xmax = 15.0

# initialize the equation and the space
eq1 = PDE({"φ": f"laplace({varH}*φ/({stoppingPower}*{scatteringRate1})) - d_dx(φ*{varH}*{scatteringRate1_dx}/{stoppingPower})"})
eq2 = PDE({"φ": f"laplace({varH}*φ/({stoppingPower}*{scatteringRate2})) - d_dx(φ*{varH}*{scatteringRate2_dx}/{stoppingPower})"})

grid = CartesianGrid([(-xmax, xmax)], [xbins], periodic=True)
state = ScalarField.from_expression(grid, f"exp(-0.5*((x-{muInit})/{sigmaInit})**2)/({sigmaInit}*sqrt(2*{np.pi}))")

# Initialize Monte Carlo Algorithm

# Set up simulation domain
ymin = -xmax; ymax = xmax; ybins = 2
zmin = -xmax; zmax = xmax; zbins = 2
simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

# Set up initial conditions
NB_PARTICLES = 30000
eSource: float = tmax
SEED: int = 4  # Random number generator seed
pointSourceSim = DiffusionPointSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource, loc=muInit, std=sigmaInit)

# Scattering angle must be independent of previous angle!
DiffusionPointSource.SAMPLE_NEW_ABSOLUTE_DIRECTION = True  # type: ignore

particle1 = DiffusionTestParticle(Es=scatteringRate1)
particle2 = DiffusionTestParticle(Es=scatteringRate2)

particleTracer1 = AnalogParticleTracer(particle=particle1, simOptions=pointSourceSim, simDomain=simDomain)
particleTracer2 = AnalogParticleTracer(particle=particle2, simOptions=pointSourceSim, simDomain=simDomain)

file1 = 'trackEndEstimatorx1.pkl'
file2 = 'trackEndEstimatorx2.pkl'

if __name__ == '__main__':
    nproc = MPI.COMM_WORLD.Get_size()
    eSource = float(sys.argv[1])
    NB_PARTICLES_PER_PROC = int(int(sys.argv[2])/nproc)
    NB_PARTICLES = NB_PARTICLES_PER_PROC*nproc

    # - Set up estimator and particle
    trackEndEstimatorx1 = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
    trackEndEstimatorx2 = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')

    # - Run simulation
    particleTracer1.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorx1, ), file=file1)
    particleTracer2.runMultiProc(nbParticles=NB_PARTICLES, estimators=(trackEndEstimatorx2, ), file=file2)
