import sys
import math
import time
from uu import Error
import numpy as np
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.Material import Material
from egsMS import mscat

if __name__ == '__main__':
    nbsamples = int(float(sys.argv[1]))

    # energy array
    minEnergy = 1e-3
    maxEnergy = 21/ERE
    nbEnergy = int(float(sys.argv[2]))
    energyArray = np.linspace(minEnergy, maxEnergy, nbEnergy)

    # stepsize array
    minds = 0.0
    maxds = 0.1
    nbds = int(float(sys.argv[3]))
    stepsizeArray = np.linspace(minds, maxds, nbds)

    # material array. Specific for lung test case.
    # minDensity = 0.05
    # maxDensity = 1.85
    # temp = maxDensity*np.arange(0, 256)/255
    # indices = temp < minDensity
    # temp[indices] = minDensity
    # uniqueDensities = np.unique(temp)
    uniqueDensities = np.linspace(1.0, 1.85, 2)

    lutMean = np.empty(shape=(nbEnergy, nbds, uniqueDensities.size, 2), dtype=float)
    lutVar = np.empty(shape=(nbEnergy, nbds, uniqueDensities.size, 2), dtype=float)

    t1 = time.process_time()

    for i, energy in enumerate(energyArray):
        t3 = time.process_time()
        for j, stepsize in enumerate(stepsizeArray):
            for k, density in enumerate(uniqueDensities):
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

                lutMean[i, j, k, :] = (meanMu, meanSint)
                lutVar[i, j, k, :] = (sosMu/(nbsamples-1), sosSint/(nbsamples-1))
        t4 = time.process_time()
        print(f'{round(100*(i+1)/nbEnergy, 3)}% completed. Last section took {(t4-t3)/60:2e} minutes.')

    t2 = time.process_time()

    with open('lutMean.npy', 'wb') as f:
        np.save(f, lutMean)
    with open('lutVar.npy', 'wb') as f:
        np.save(f, lutVar)

    print(f'Total time: {(t2-t1)/60:2e} minutes.')