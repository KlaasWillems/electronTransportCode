# Add root directory to path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Imports
from electronTransportCode.SimOptions import LineSourceSimulation
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.MCEstimator import FluenceEstimator
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.ParticleModel import LineSourceParticle
from electronTransportCode.ProjectUtils import E_THRESHOLD
from electronTransportCode.Material import unitDensityMaterial
import time

# Set up initial conditions
eSource: float = 1.0
SEED: int = 4  # Random number generator seed
lineSourceSim = LineSourceSimulation(minEnergy=0, eSource=eSource, rngSeed=SEED)

# Set up simulation domain
xmin = -1.5; xmax = 1.5; xbins = 200
simDomain = SimulationDomain(xmin, xmax, xmin, xmax, xbins, xbins, material=unitDensityMaterial)

# Set up dose estimator
Ebins = 20
fluenceEstimator = FluenceEstimator(simDomain=simDomain, Emin=0.0, Emax=eSource, Ebins=Ebins)

# Set up particle
particle = LineSourceParticle(generator=SEED)  # rng is later overridden by simulation object

# Set up particle tracer
particleTracer = AnalogParticleTracer(particle=particle, simOptions=lineSourceSim, simDomain=simDomain)

if __name__ == '__main__':
    NB_PARTICLES = 10000
    t1 = time.perf_counter()
    particleTracer(nbParticles=NB_PARTICLES, estimator=fluenceEstimator)
    t2 = time.perf_counter()
    print(f'Average amount of events: {particleTracer.averageNbCollisions}')