import time
import pickle
from electronTransportCode.SimOptions import PointSourceSimulation
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.MCEstimator import FluenceEstimator, DoseEstimator
from electronTransportCode.ParticleModel import PointSourceParticle
from electronTransportCode.Material import unitDensityMaterial

# Set up initial conditions
eSource: float = 1.0
SEED: int = 4  # Random number generator seed
lineSourceSim = PointSourceSimulation(minEnergy=0, eSource=eSource, rngSeed=SEED)

# Set up simulation domain
xmin = -1.0; xmax = 1.0; xbins = 100
simDomain = SimulationDomain(xmin, xmax, xmin, xmax, xbins, xbins, material=unitDensityMaterial)

# Set up dose estimator
Ebins = 100
fluenceEstimator = FluenceEstimator(simDomain=simDomain, Emin=0.0, Emax=eSource, Ebins=Ebins)
doseEstimator = DoseEstimator(simDomain)

# Set up particle
particle = PointSourceParticle(generator=SEED)  # rng is later overridden by simulation object

# Set up particle tracer
particleTracer = AnalogParticleTracer(particle=particle, simOptions=lineSourceSim, simDomain=simDomain)

if __name__ == '__main__':
    NB_PARTICLES = 1000000
    t1 = time.perf_counter()
    particleTracer(nbParticles=NB_PARTICLES, estimators=(fluenceEstimator, doseEstimator))
    t2 = time.perf_counter()
    print(f'Average amount of events per particle: {particleTracer.averageNbCollisions}')
    print(f'Simulation took {(t2-t1)/60} minutes')

    # Save files
    with open('data/fluenceEstimator.pkl', 'wb') as file:
        pickle.dump(fluenceEstimator, file)

    with open('data/particleTracer.pkl', 'wb') as file:
        pickle.dump(particleTracer, file)

    with open('data/doseEstimator.pkl', 'wb') as file:
        pickle.dump(doseEstimator, file)