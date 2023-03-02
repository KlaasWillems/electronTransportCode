import time
import pickle
from electronTransportCode.SimOptions import PointSourceSimulation
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.MCParticleTracer import AnalogParticleTracer
from electronTransportCode.MCEstimator import FluenceEstimator, DoseEstimator, TrackEndEstimator
from electronTransportCode.ParticleModel import PointSourceParticle
from electronTransportCode.Material import unitDensityMaterial

# Set up simulation domain
xmin = -1.1; xmax = 1.1; xbins = 100
ymin = -1.1; ymax = 1.1; ybins = 100
simDomain = SimulationDomain(xmin, xmax, ymin, ymax, xbins, ybins, material=unitDensityMaterial)

# Set up initial conditions
NB_PARTICLES = 500000
eSource: float = 1.0
SEED: int = 4  # Random number generator seed
pointSourceSim = pointSourceSim = PointSourceSimulation(minEnergy=0.0, rngSeed=SEED, eSource=eSource)

# Set up dose estimator
Ebins = 100
fluenceEstimator = FluenceEstimator(simDomain=simDomain, Emin=0.0, Emax=eSource, Ebins=Ebins)
doseEstimator = DoseEstimator(simDomain)
trackEndEstimator = TrackEndEstimator(simDomain, NB_PARTICLES)

# Set up particle
particle = PointSourceParticle(generator=SEED)  # rng is later overridden by simulation object

# Set up particle tracer
particleTracer = AnalogParticleTracer(particle=particle, simOptions=pointSourceSim, simDomain=simDomain)

if __name__ == '__main__':
    t1 = time.perf_counter()
    particleTracer(nbParticles=NB_PARTICLES, estimators=(fluenceEstimator, doseEstimator, trackEndEstimator))
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

    with open('data/trackEndEstimator.pkl', 'wb') as file:
        pickle.dump(trackEndEstimator, file)
