import sys
import os
import math
import pickle
import time
from typing import Optional, Tuple
import numpy as np
from mpi4py import MPI
from spherical_stats import _vmf
from electronTransportCode.SimOptions import KDRLUTSimulation
from electronTransportCode.MCEstimator import TrackEndEstimator
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron, KDRTestParticle, ParticleModel
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.Material import Material

# Make LUT for multiple scattering angle theta and phi. LUT stores distribution of theta and phi as histogram.
# Command line arguments:
#   1) Amount of particles to simulate per setting

xmin = -100
xmax = 100
xbins = ybins = 1
tempFile = 'data/tempTEE.pkl'

def sample(NB_PARTICLES: int, NB_PARTICLES_PER_PROC: int, particle: ParticleModel, energy: float, stepsize: float, material: Material) -> Tuple[TrackEndEstimator, TrackEndEstimator, TrackEndEstimator]:
    simDomain = SimulationDomain(xmin, xmax, xmin, xmax, xbins, ybins, material)
    deltaE = particle.energyLoss(energy, np.array((0, 0, 0), dtype=float), stepsize, material)
    assert energy-deltaE >= 0, f'{energy=}, {deltaE=}'
    simOptions = KDRLUTSimulation(eSource=energy, minEnergy=energy-deltaE, rngSeed=MPI.COMM_WORLD.Get_rank())
    TEEx = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'Omega_x')
    TEEy = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'Omega_y')
    TEEz = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'Omega_z')
    particleTracer = AnalogParticleTracer(simOptions, simDomain, particle)
    particleTracer.runMultiProc(NB_PARTICLES, (TEEx, TEEy, TEEz), file=None, verbose=False)
    return TEEx, TEEy, TEEz

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()
    NB_PARTICLES_PER_PROC = int(float(sys.argv[1])/nproc)
    NB_PARTICLES = int(NB_PARTICLES_PER_PROC*nproc)

    particle = SimplifiedEGSnrcElectron()
    energyArray = np.array((6.1, 6.1, 20, 20), dtype=float)
    stepsizeArray = np.array((0.01, 0.1, 0.01, 0.1), dtype=float)
    densityArray = np.ones_like(energyArray)

    lut: Optional[np.ndarray]
    if rank == 0:
        lut = np.empty(shape=(energyArray.size, NB_PARTICLES, 3), dtype=float)  # Store fitted dipserion coefficient
    else:
        lut = None

    t1 = time.process_time()

    # Do simulation for each combination of energy, stepsize and material
    for i, (energy, stepsize, density) in enumerate(zip(energyArray, stepsizeArray, densityArray)):
        t3 = time.process_time()
        TEEx: TrackEndEstimator; TEEy: TrackEndEstimator; TEEz: TrackEndEstimator
        material = Material(rho=density)
        TEEx, TEEy, TEEz = sample(NB_PARTICLES, NB_PARTICLES_PER_PROC, particle, energy, stepsize, material)
        if rank == 0:  # post-process simulation results
            assert lut is not None
            lut[i, :, 0] = TEEx.scoreMatrix
            lut[i, :, 1] = TEEy.scoreMatrix
            lut[i, :, 2] = TEEz.scoreMatrix
        t4 = time.process_time()
        if rank == 0:
            print(f'{round(100*(i+1)/energyArray.size, 3)}% completed. Last section took {(t4-t3)/60:2e} minutes.')

    if rank == 0:
        assert lut is not None
        np.save('data/msvelocityDist.npy', lut)
        np.savez('data/msvelocityDist.npz', energyArray, stepsizeArray, densityArray)
        t2 = time.process_time()
        print(f'Total time: {(t2-t1)/60:2e} minutes.')
