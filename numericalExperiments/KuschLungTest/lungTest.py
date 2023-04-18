import sys
import pickle
from mpi4py import MPI
from electronTransportCode.MCEstimator import DoseEstimator
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from lungSetup import LungInitialConditions, LungSimulationDomain

# Load particle, initial conditions and simulation domain

if __name__ == '__main__':

    # Sim settings
    particle = SimplifiedEGSnrcElectron(scatterer='2d')  # constrain scattering to yz plance
    lungInit = LungInitialConditions(sigmaPos=1, kappa=10)
    lungSimDomain = LungSimulationDomain()

    # Load particle tracers
    analogTracer = AnalogParticleTracer(lungInit, lungSimDomain, particle)

    # Estimator
    doseEstimatorK = DoseEstimator(lungSimDomain)

    # MPI Code
    procs = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    NB_PARTICLES = int(sys.argv[1])
    NB_PARTICLES_PER_PROC = int(NB_PARTICLES/procs)
    NB_PARTICLES = int(procs*NB_PARTICLES_PER_PROC)

    analogTracer.runMultiProc(NB_PARTICLES, (doseEstimatorK, ), file='data/doseEstimatorK.pkl')

    pickle.dump(analogTracer, open(f'data/analogTracer.pkl', mode='wb'))
