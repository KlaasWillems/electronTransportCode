import sys
import math
from typing import Optional
import time
import numpy as np
from mpi4py import MPI
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.Material import Material
from egsMS import mscat

if __name__ == '__main__':
    nbsamples = int(float(sys.argv[1]))
    nproc = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    nbEnergy = int(float(sys.argv[2]))

    if nbEnergy < nproc:
        raise ValueError('More processors than energy values.')

    # energy array
    minEnergy = 1e-3
    maxEnergy = 21/ERE
    energies = np.linspace(minEnergy, maxEnergy, nbEnergy)
    energiesSplit = np.array_split(energies, nproc)
    energyArray = energiesSplit[rank]

    # stepsize array
    minds = 0.0
    maxds = 0.1
    nbds = int(float(sys.argv[3]))
    stepsizeArray = np.linspace(minds, maxds, nbds)

    # material array. Specific for lung test case.
    minDensity = 0.05
    maxDensity = 1.85
    nbDensity = int(float(sys.argv[4]))
    densityArray = np.linspace(minDensity, maxDensity, nbDensity)

    lut = np.empty(shape=(energyArray.size, nbds, nbDensity, 4), dtype=float)

    t1 = time.process_time()

    for i, energy in enumerate(energyArray):
        t3 = time.process_time()
        for j, stepsize in enumerate(stepsizeArray):
            for k, density in enumerate(densityArray):
                material = Material(rho=density)

                # sum of squares and mean for mu and sint
                meanMu = 0.0
                sosMu = 0.0
                meanSint = 0.0
                sosSint = 0.0

                for sampleNb in range(1, nbsamples+1):
                    # sample cos(theta)
                    mu = mscat(energy, stepsize, material.Z, material.eta0CONST, material.bc)
                    sint = math.sqrt(1.0 - mu**2)

                    # update mean and variance
                    dmu = mu - meanMu
                    dsint = sint - meanSint
                    meanMu = meanMu + dmu/sampleNb
                    meanSint = meanSint + dsint/sampleNb
                    sosMu = sosMu + dmu*(mu - meanMu)
                    sosSint = sosSint + dsint*(sint - meanSint)

                lut[i, j, k, :] = (meanMu, meanSint, sosMu/(nbsamples-1), sosSint/(nbsamples-1))
        t4 = time.process_time()
        print(f'Proc: {rank}. {round(100*(i+1)/energyArray.size, 3)}% completed. Last section took {(t4-t3)/60:2e} minutes.')

    t2 = time.process_time()
    bigLut: Optional[np.ndarray]
    if nproc > 1:
        if rank == 0:
            bigLut = np.empty(shape=(nbEnergy, nbds, nbDensity, 4), dtype=float)
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
        np.save('data/lut.npy', bigLut)
        np.savez('data/lutAxes.npz', energies, stepsizeArray, densityArray)

        print(f'Total time: {(t2-t1)/60:2e} minutes.')