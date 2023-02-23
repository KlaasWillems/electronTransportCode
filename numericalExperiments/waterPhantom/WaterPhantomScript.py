import time
import pickle
from electronTransportCode.SimOptions import WaterPhantomSimulation
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.MCEstimator import DoseEstimator
from electronTransportCode.ProjectUtils import ERE
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron
from electronTransportCode.ProjectUtils import E_THRESHOLD
from electronTransportCode.Material import WaterMaterial

# Set up initial conditions
eInit: float = 5.0/ERE  # 5 MeV initial energy
SEED: int = 4  # Random number generator seed
xVariance: float = 0.1  # Variance on initial position in x and y direction
waterPhantomInit = WaterPhantomSimulation(minEnergy=E_THRESHOLD, eSource=eInit, xVariance=xVariance, rngSeed=SEED)

# Set up simulation domain
simDomain = SimulationDomain(-2.5, 7.5, -2.5, 7.5, 200, 200, material=WaterMaterial)

# Set up dose estimator
doseEstimator = DoseEstimator(simDomain=simDomain)

# Set up particle
particle = SimplifiedEGSnrcElectron(generator=None)  # rng is later added by simulation object

# Set up particle tracer
particleTracer = AnalogParticleTracer(particle=particle, simOptions=waterPhantomInit, simDomain=simDomain)

if __name__ == "__main__":
    # Run simulation
    print('Starting simulation')
    NB_PARTICLES = 10000
    t1 = time.perf_counter()
    particleTracer(nbParticles=NB_PARTICLES, estimator=doseEstimator)
    t2 = time.perf_counter()
    print(f'Simulation took {t2-t1} seconds')

    # Save files
    with open('data/doseEstimator.pkl', 'wb') as file:
        pickle.dump(doseEstimator, file)

    with open('data/particleTracer.pkl', 'wb') as file:
        pickle.dump(particleTracer, file)
