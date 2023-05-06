import sys
import os
import math
import pickle
import time
from typing import Optional
import numpy as np
from mpi4py import MPI
from electronTransportCode.SimOptions import KDRLUTSimulation
from electronTransportCode.MCEstimator import TrackEndEstimator
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.ParticleModel import KDRTestParticle, SimplifiedEGSnrcElectron
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.Material import Material, WaterMaterial

xmin = -100
xmax = 100
xbins = ybins = 1

if __name__ == '__main__':
    # sim options
    NB_PARTICLES = 1
    energy = 10
    stepsize = 16*0.1
    material = WaterMaterial

    # set up sim
    simDomain = SimulationDomain(xmin, xmax, xmin, xmax, xbins, ybins, material)
    particle = KDRTestParticle()
    deltaE = particle.energyLoss(energy, np.array((0, 0, 0), dtype=float), stepsize, material)
    print(particle.getScatteringRate(None, deltaE, WaterMaterial))
    assert energy-deltaE >= 0, f'{energy=}, {deltaE=}'
    simOptions = KDRLUTSimulation(eSource=energy, minEnergy=energy-deltaE, rngSeed=MPI.COMM_WORLD.Get_rank())
    TEEx = TrackEndEstimator(simDomain, NB_PARTICLES, 'x')
    TEEy = TrackEndEstimator(simDomain, NB_PARTICLES, 'y')
    TEEz = TrackEndEstimator(simDomain, NB_PARTICLES, 'z')
    particleTracer = AnalogParticleTracer(simOptions, simDomain, particle)
    particleTracer.__call__(1, (TEEx, TEEy, TEEz))
    # print(particleTracer.pathlength)
