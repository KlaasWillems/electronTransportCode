import sys
import math
from typing import Optional
import time
import numpy as np
from mpi4py import MPI
from electronTransportCode.SimOptions import KDRLUTSimulation
from electronTransportCode.MCEstimator import TrackEndEstimator
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.ParticleModel import KDRTestParticle
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.Material import Material

xmin = -100
xmax = 100
xbins = ybins = 1

def sample(nbsamples: int, energy: float, stepsize: float, material: Material) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    simDomain = SimulationDomain(xmin, xmax, xmin, xmax, xbins, ybins, material)
    particle = KDRTestParticle(None)
    deltaE = particle.energyLoss(energy, np.array((0, 0, 0), dtype=float), stepsize, material)
    if energy > deltaE:
        # assert energy > deltaE, f'{energy=}, {deltaE=}, {stepsize=}. Stepsize too large for available energy'
        simOptions = KDRLUTSimulation(eSource=energy, minEnergy=energy-deltaE, rngSeed=MPI.COMM_WORLD.Get_rank())
        TEEx = TrackEndEstimator(simDomain, nbsamples, 'x')
        TEEy = TrackEndEstimator(simDomain, nbsamples, 'y')
        TEEz = TrackEndEstimator(simDomain, nbsamples, 'z')
        particleTracer = AnalogParticleTracer(simOptions, simDomain, particle)
        particleTracer.__call__(nbsamples, (TEEx, TEEy, TEEz))

        temp = np.zeros_like(TEEx.scoreMatrix)  # cos(theta) array
        for i in range(nbsamples):
            x = TEEx.scoreMatrix[i]
            y = TEEy.scoreMatrix[i]
            z = TEEz.scoreMatrix[i]
            temp[i] = z/math.sqrt(x**2 + y**2 + z**2)

        return temp.mean(), TEEx.scoreMatrix.mean(), TEEy.scoreMatrix.mean(), TEEz.scoreMatrix.mean(), TEEx.scoreMatrix.var(), TEEy.scoreMatrix.var(), TEEz.scoreMatrix.var()
    else:
        print(f'{energy=}, {deltaE=}, {stepsize=}')
        return None, None, None, None, None, None, None

if __name__ == '__main__':
    nbsamples = int(float(sys.argv[1]))
    nproc = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    nbEnergy = int(float(sys.argv[2]))

    if nbEnergy < nproc:
        raise ValueError('More processors than energy values.')

    # energy array
    minEnergy = 5e-1
    maxEnergy = 21/ERE
    energies = np.linspace(minEnergy, maxEnergy, nbEnergy)
    energiesSplit = np.array_split(energies, nproc)
    energyArray = energiesSplit[rank]

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

    lut = np.empty(shape=(energyArray.size, nbds, nbDensity, 7), dtype=float)

    t1 = time.process_time()

    for i, energy in enumerate(energyArray):
        t3 = time.process_time()
        for j, stepsize in enumerate(stepsizeArray):
            for k, density in enumerate(densityArray):
                material = Material(rho=density)
                lut[i, j, k, :] = sample(nbsamples, energy, stepsize, material)
        t4 = time.process_time()
        print(f'Proc: {rank}. {round(100*(i+1)/energyArray.size, 3)}% completed. Last section took {(t4-t3)/60:2e} minutes.')

    t2 = time.process_time()
    bigLut: Optional[np.ndarray]
    if nproc > 1:
        if rank == 0:
            bigLut = np.empty(shape=(nbEnergy, nbds, nbDensity, 7), dtype=float)
            beginIndex = 0
            for index, energyGroup in enumerate(energiesSplit):
                if index != 0:
                    MPI.COMM_WORLD.Recv(bigLut[beginIndex:, :, :, :], source=index, tag=index)
                else:
                    bigLut[beginIndex:beginIndex+energyArray.size, :, :, :] = lut
                beginIndex += energyArray.size

        else:
            MPI.COMM_WORLD.Send(lut, dest=0, tag=rank)
            bigLut = None
    else:
        bigLut = lut

    if rank == 0:
        assert bigLut is not None
        np.save('data/ownlut.npy', bigLut)
        np.savez('data/ownlutAxes.npz', energies, stepsizeArray, densityArray)

        print(f'Total time: {(t2-t1)/60:2e} minutes.')