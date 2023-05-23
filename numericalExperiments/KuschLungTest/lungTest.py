import sys
import pickle
import time
import numpy as np
from mpi4py import MPI
from electronTransportCode.MCEstimator import DoseEstimator
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron, KDRTestParticle, ParticleModel
from electronTransportCode.MCParticleTracer import AnalogParticleTracer, KDMC, KDR
from electronTransportCode.ProjectUtils import ERE
from lungSetup import LungInitialConditions, LungSimulationDomain

# Load particle, initial conditions and simulation domain

if __name__ == '__main__':
    NB_PARTICLES = int(sys.argv[1])
    scatterer = sys.argv[2]
    algorithm = sys.argv[3]
    particleType = sys.argv[4]

    # Sim settings
    particle: ParticleModel
    if particleType == 'EGS':
        particle = SimplifiedEGSnrcElectron(scatterer=scatterer)  # constrain scattering to yz plance
    elif particleType == 'KDRTest':
        particle = KDRTestParticle(msDist='vmf')
    else:
        raise ValueError
    lungInit = LungInitialConditions(sigmaPos=1/50, kappa=10, eSource=35/ERE)
    lungSimDomain = LungSimulationDomain()

    if scatterer == '2d-simple':  # decrease scattering rate for easy testing
        for material in lungSimDomain.materialArray:
            material.SigmaCONST = material.SigmaCONST/100

    # MPI Code
    procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/procs)
    NB_PARTICLES = int(procs*NB_PARTICLES_PER_PROC)
    _, step = np.linspace(0, lungSimDomain.width, lungSimDomain.bins+1, retstep=True)

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
        kdmc = KDMC(lungInit, lungSimDomain, particle, dS=step)  # type: ignore

        # Estimator
        doseEstimatorKD = DoseEstimator(lungSimDomain)
        t1 = time.process_time()
        kdmc.runMultiProc(NB_PARTICLES, (doseEstimatorKD, ), file=f'data/doseEstimatorKD{scatterer}.pkl', logAmount=500)
        t2 = time.process_time()
        pickle.dump(kdmc, open(f'data/kdmc{scatterer}.pkl', mode='wb'))
    elif algorithm == 'kdr':
        kdr = KDR(simOptions=lungInit, simDomain=lungSimDomain, particle=particle, dS=step, useMSAngle=False)  # type: ignore

        # Estimator
        doseEstimatorKD = DoseEstimator(lungSimDomain)
        t1 = time.process_time()
        kdr.runMultiProc(NB_PARTICLES, (doseEstimatorKD, ), file=f'data/doseEstimatorKDR{scatterer}.pkl', logAmount=500)
        t2 = time.process_time()
        pickle.dump(kdr, open(f'data/kdr{scatterer}.pkl', mode='wb'))
    elif algorithm == 'kdrMS':
        kdr = KDR(simOptions=lungInit, simDomain=lungSimDomain, particle=particle, dS=step, useMSAngle=True)  # type: ignore

        # Estimator
        doseEstimatorKD = DoseEstimator(lungSimDomain)
        t1 = time.process_time()
        kdr.runMultiProc(NB_PARTICLES, (doseEstimatorKD, ), file=f'data/doseEstimatorKDRMS{scatterer}.pkl', logAmount=500)
        t2 = time.process_time()
        pickle.dump(kdr, open(f'data/kdrMS{scatterer}.pkl', mode='wb'))

    else:
        raise ValueError('Wrong algorithm input.')

    print(f'Process {rank}: Simulation took: {(t2-t1)/60:.3e} minutes')