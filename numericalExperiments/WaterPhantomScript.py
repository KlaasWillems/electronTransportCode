# Add root directory to path
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# Imports
from electronTransportCode.SimOptions import WaterPhantomSimulation
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.MCEstimator import DoseEstimator
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron
from electronTransportCode.ProjectUtils import E_THRESHOLD

# Set up initial conditions
eInit: float = 5/ERE  # 5 MeV initial energy
energyCutOff: float = E_THRESHOLD
SEED: int = 4  # Random number generator seed
xVariance: float = 0.1  # Variance on initial position in x and y direction
waterPhantomInit = WaterPhantomSimulation(minEnergy=energyCutOff, Esource=eInit, xVariance=xVariance, rngSeed=SEED)

# Set up simulation domain
simDomain = SimulationDomain(-2.5, 7.5, -2.5, 7.5, 200, 200)

# Set up dose estimator
doseEstimator = DoseEstimator(simDomain=simDomain)

# Set up particle
particle = SimplifiedEGSnrcElectron(generator=None)  # rng is later added by simulation object

# Set up particle tracer
particleTracer = AnalogParticleTracer(particle=particle, simOptions=waterPhantomInit, simDomain=simDomain)

if __name__ == "__main__":
    # Run simulation
    NB_PARTICLES = 30
    particleTracer(nbParticles=NB_PARTICLES, estimator=doseEstimator)
    print(f'Average amount of events: {particleTracer.averageNbCollisions}')
