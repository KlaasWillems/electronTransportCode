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
from electronTransportCode.Material import Material

# Make LUT for mean and variance for KDR.

xmin = -100
xmax = 100
xbins = ybins = 1
tempFile = 'data/tempTEE.pkl'

def sample(NB_PARTICLES: int, NB_PARTICLES_PER_PROC: int, energy: float, stepsize: float, material: Material) -> None:
    simDomain = SimulationDomain(xmin, xmax, xmin, xmax, xbins, ybins, material)
    particle = SimplifiedEGSnrcElectron()
    deltaE = particle.energyLoss(energy, np.array((0, 0, 0), dtype=float), stepsize, material)
    assert energy-deltaE >= 0, f'{energy=}, {deltaE=}'
    simOptions = KDRLUTSimulation(eSource=energy, minEnergy=energy-deltaE, rngSeed=MPI.COMM_WORLD.Get_rank())
    TEEx = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'x')
    TEEy = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'y')
    TEEz = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, 'z')
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
        lut = np.empty(shape=(nbEnergy, nbds, nbDensity, 7), dtype=float)
    else:
        lut = None

    t1 = time.process_time()

    # Do simulation for each combination of energy, stepsize and material
    for i, energy in enumerate(energyArray):
        t3 = time.process_time()
        for j, stepsize in enumerate(stepsizeArray):
            for k, density in enumerate(densityArray):
                material = Material(rho=density)
                sample(NB_PARTICLES, NB_PARTICLES_PER_PROC, energy, stepsize, material)
                if rank == 0:
                    assert lut is not None
                    TEEx, TEEy, TEEz = pickle.load(open(tempFile, 'rb'))
                    temp = np.zeros_like(TEEx.scoreMatrix)  # cos(theta) array
                    for l in range(NB_PARTICLES):
                        x = TEEx.scoreMatrix[l]
                        y = TEEy.scoreMatrix[l]
                        z = TEEz.scoreMatrix[l]
                        temp[l] = z/math.sqrt(x**2 + y**2 + z**2)
                    lut[i, j, k, :] = temp.mean(), TEEx.scoreMatrix.mean(), TEEy.scoreMatrix.mean(), TEEz.scoreMatrix.mean(), TEEx.scoreMatrix.var(), TEEy.scoreMatrix.var(), TEEz.scoreMatrix.var()
        t4 = time.process_time()
        if rank == 0:
            print(f'{round(100*(i+1)/stepsizeArray.size, 3)}% completed. Last section took {(t4-t3)/60:2e} minutes.')

    if rank == 0:
        assert lut is not None
        os.remove(tempFile)
        np.save('data/ownlutv2.npy', lut)
        np.savez('data/ownlutAxesv2.npz', energyArray, stepsizeArray, densityArray)
        t2 = time.process_time()
        print(f'Total time: {(t2-t1)/60:2e} minutes.')
