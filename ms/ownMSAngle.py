import sys
import os
import math
import pickle
import time
from typing import Optional
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
#   2) Amount of energy bins. If 1, simulate KDRTestParticle that has fixed energy.
#   3) Amount of stepsize bins
#   4) Amount of density bins
#   5) Amount of bins in de histogram

xmin = -100
xmax = 100
xbins = ybins = 1
tempFile = 'data/tempTEE.pkl'

def sample(NB_PARTICLES: int, NB_PARTICLES_PER_PROC: int, particle: ParticleModel, energy: float, stepsize: float, material: Material) -> None:
    simDomain = SimulationDomain(xmin, xmax, xmin, xmax, xbins, ybins, material)
    deltaE = particle.energyLoss(energy, np.array((0, 0, 0), dtype=float), stepsize, material)
    assert energy-deltaE >= 0, f'{energy=}, {deltaE=}'
    simOptions = KDRLUTSimulation(eSource=energy, minEnergy=energy-deltaE, rngSeed=MPI.COMM_WORLD.Get_rank())
    TEEx = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'Omega_x')
    TEEy = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'Omega_y')
    TEEz = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'Omega_z')
    particleTracer = AnalogParticleTracer(simOptions, simDomain, particle)
    particleTracer.runMultiProc(NB_PARTICLES, (TEEx, TEEy, TEEz), file=tempFile, verbose=False)

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()
    NB_PARTICLES_PER_PROC = int(float(sys.argv[1])/nproc)
    NB_PARTICLES = int(NB_PARTICLES_PER_PROC*nproc)

    # energy array
    minEnergy = 5e-1
    maxEnergy = 21/ERE
    nbEnergy = int(float(sys.argv[2]))
    particle: ParticleModel
    if nbEnergy == 1:
        particle = KDRTestParticle()
        energyArray = np.array((particle.EFixed, ), dtype=float)
    else:
        particle = SimplifiedEGSnrcElectron()
        energyArray = np.linspace(minEnergy, maxEnergy, nbEnergy)

    # stepsize array
    minds = 1e-4
    maxds = 0.1
    nbds = int(float(sys.argv[3]))
    stepsizeArray = np.linspace(minds, maxds, nbds)

    # material array. Specific for lung test case.
    minDensity = 0.05
    maxDensity = 1.85
    nbDensity = int(float(sys.argv[4]))
    densityArray = np.linspace(minDensity, maxDensity, nbDensity)

    lut: Optional[np.ndarray]
    if rank == 0:
        lut = np.empty(shape=(nbEnergy, nbds, nbDensity, 1), dtype=float)  # Store fitted dipserion coefficient
    else:
        lut = None

    t1 = time.process_time()

    # Do simulation for each combination of energy, stepsize and material
    for i, energy in enumerate(energyArray):
        t3 = time.process_time()
        for j, stepsize in enumerate(stepsizeArray):
            for k, density in enumerate(densityArray):
                material = Material(rho=density)
                sample(NB_PARTICLES, NB_PARTICLES_PER_PROC, particle, energy, stepsize, material)
                if rank == 0:  # post-process simulation results
                    assert lut is not None
                    TEEx: TrackEndEstimator; TEEy: TrackEndEstimator; TEEz: TrackEndEstimator
                    TEEx, TEEy, TEEz = pickle.load(open(tempFile, 'rb'))
                    data = np.column_stack((TEEx.scoreMatrix, TEEy.scoreMatrix, TEEz.scoreMatrix))
                    mean, kappa = _vmf._fit(data)
                    lut[i, j, k, 0] = kappa
        t4 = time.process_time()
        if rank == 0:
            print(f'{round(100*(i+1)/energyArray.size, 3)}% completed. Last section took {(t4-t3)/60:2e} minutes.')

    if rank == 0:
        assert lut is not None
        os.remove(tempFile)
        np.save('data/msKappaLUT.npy', lut)
        np.savez('data/msKappaAxes.npz', energyArray, stepsizeArray, densityArray)
        t2 = time.process_time()
        print(f'Total time: {(t2-t1)/60:2e} minutes.')
