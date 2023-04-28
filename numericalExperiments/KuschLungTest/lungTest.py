import sys
import pickle
import time
import numpy as np
from mpi4py import MPI
from electronTransportCode.MCEstimator import DoseEstimator
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron
from electronTransportCode.MCParticleTracer import AnalogParticleTracer, KDMC, KDR
from lungSetup import LungInitialConditions, LungSimulationDomain

# Load particle, initial conditions and simulation domain

if __name__ == '__main__':
    NB_PARTICLES = int(sys.argv[1])
    scatterer = sys.argv[2]
    algorithm = sys.argv[3]

    # Sim settings
    particle = SimplifiedEGSnrcElectron(scatterer=scatterer)  # constrain scattering to yz plance
    lungInit = LungInitialConditions(sigmaPos=1, kappa=10)
    lungSimDomain = LungSimulationDomain()

    if scatterer == '2d-simple':  # decrease scattering rate for easy testing
        for material in lungSimDomain.materialArray:
            material.bc = material.bc/100

    # MPI Code
    procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/procs)
    NB_PARTICLES = int(procs*NB_PARTICLES_PER_PROC)

    if algorithm == 'k':
        # Load particle tracers
        analogTracer = AnalogParticleTracer(lungInit, lungSimDomain, particle)

        # Estimator
        doseEstimatorK = DoseEstimator(lungSimDomain)
        t1 = time.process_time()
        analogTracer.runMultiProc(NB_PARTICLES, (doseEstimatorK, ), file=f'data/doseEstimatorK{scatterer}.pkl', logAmount=500)
        t2 = time.process_time()
        pickle.dump(analogTracer, open(f'data/analogTracer{scatterer}.pkl', mode='wb'))
    elif algorithm == 'kd':
        # Load particle tracers
        _, step = np.linspace(0, lungSimDomain.width, lungSimDomain.bins+1, retstep=True)
        kdmc = KDMC(lungInit, lungSimDomain, particle, dS=step)  # type: ignore

        # Estimator
        doseEstimatorKD = DoseEstimator(lungSimDomain)
        t1 = time.process_time()
        kdmc.runMultiProc(NB_PARTICLES, (doseEstimatorKD, ), file=f'data/doseEstimatorKD{scatterer}.pkl', logAmount=500)
        t2 = time.process_time()
        pickle.dump(kdmc, open(f'data/kdmc{scatterer}.pkl', mode='wb'))
    elif algorithm == 'kdr':
        _, step = np.linspace(0, lungSimDomain.width, lungSimDomain.bins+1, retstep=True)
        kdr = KDR(simOptions=lungInit, simDomain=lungSimDomain, particle=particle, dS=step)  # type: ignore

        # Estimator
        doseEstimatorKD = DoseEstimator(lungSimDomain)
        t1 = time.process_time()
        kdr.runMultiProc(NB_PARTICLES, (doseEstimatorKD, ), file=f'data/doseEstimatorKDR{scatterer}.pkl', logAmount=500)
        t2 = time.process_time()
        pickle.dump(kdr, open(f'data/kdr{scatterer}.pkl', mode='wb'))
    else:
        raise ValueError('Wrong algorithm input.')

    print(f'Process {rank}: Simulation took: {(t2-t1)/60:.3e} minutes')