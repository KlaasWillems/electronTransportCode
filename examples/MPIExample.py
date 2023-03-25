from mpi4py import MPI
import numpy as np
import sys
import time
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField
import matplotlib.pyplot as plt
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.Material import unitDensityMaterial
from electronTransportCode.MCEstimator import TrackEndEstimator
from electronTransportCode.ParticleModel import DiffusionTestParticle
from electronTransportCode.SimOptions import DiffusionPointSource
from electronTransportCode.MCParticleTracer import AnalogParticleTracer

# Example run:
# mpiexec -n 2 python3 -m MPIExample 1.0 10

def solvePDE(tmax: float, muInit, sigmaInit, stoppingPower, scatteringRate1, scatteringRate1_dx, varH) -> MemoryStorage:
    # initialize the equation and the space
    eq1 = PDE({"φ": f"laplace({varH}*φ/({stoppingPower}*{scatteringRate1})) - d_dx(φ*{varH}*{scatteringRate1_dx}/{stoppingPower})"})
    grid = CartesianGrid([(-xmax, xmax)], [xbins], periodic=True)
    state = ScalarField.from_expression(grid, f"exp(-0.5*((x-{muInit})/{sigmaInit})**2)/({sigmaInit}*sqrt(2*{np.pi}))")

    # solve the equation and store the trajectory
    storage1 = MemoryStorage()
    eq1.solve(state, t_range=tmax, tracker=storage1.tracker(1), dt=1e-5)
    return storage1

if __name__ == "__main__":

    t1 = time.perf_counter()

    # Initialize Advection-Diffusion parameters
    stoppingPower = 1
    scatteringRate1 = '(100 + 10*sin(x))'; scatteringRate1_dx = '(cos(x)/(10*(10 + sin(x))**2))'
    varH = 1/3  # 1 in 1D, 1/3 in 3D
    sigmaInit = 1.0
    muInit = 0.0

    # Initialize simulation parameters
    xbins = 512
    xmax = 15.0
    ymin = -xmax; ymax = xmax; ybins = 2
    zmin = -xmax; zmax = xmax; zbins = 2

    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    nproc = comm.Get_size()

    eSource = float(sys.argv[1])
    NB_PARTICLES_PER_PROC = int(int(sys.argv[2])/nproc)
    NB_PARTICLES = int(NB_PARTICLES_PER_PROC*nproc)

    if myrank == 0:
        storage1 = solvePDE(eSource, muInit, sigmaInit, stoppingPower, scatteringRate1, scatteringRate1_dx, varH)

    # --- Set up objects ---

    # Set up simulation domain
    simDomain = SimulationDomain(ymin, ymax, zmin, zmax, ybins, zbins, material=unitDensityMaterial)

    # Set up initial conditions
    SEED: int = 2 + myrank
    pointSourceSim = DiffusionPointSource(minEnergy=0.0, rngSeed=SEED, eSource=eSource, loc=muInit, std=sigmaInit)

    # Set up estimator and particle
    trackEndEstimatorx1 = TrackEndEstimator(simDomain, NB_PARTICLES_PER_PROC, setting='x')
    particle1 = DiffusionTestParticle(Es=scatteringRate1)
    particleTracer1 = AnalogParticleTracer(particle=particle1, simOptions=pointSourceSim, simDomain=simDomain)

    print(f'Proc {myrank} starting simulation.')

    # --- Run simulation ---
    particleTracer1(nbParticles=NB_PARTICLES_PER_PROC, estimators=trackEndEstimatorx1, logAmount=1000000)

    # gather estimator
    if myrank == 0:
        recvbuf = np.empty((nproc, NB_PARTICLES_PER_PROC), dtype=float)
    else:
        recvbuf = None  # type: ignore

    comm.Gather(trackEndEstimatorx1.scoreMatrix, recvbuf, root=0)

    # --- Plot result ---
    if myrank == 0:
        assert recvbuf is not None

        # Advection-diffusion solution 1
        ADres1 = storage1.data[-1]  # type:ignore
        xres = np.linspace(-xmax, xmax, xbins, endpoint=False)

        # MC solution 1
        xdensity1 = recvbuf.reshape((NB_PARTICLES, ))
        binVal1, binEdge1 = np.histogram(xdensity1, bins=100, density=True)
        binCenter1 = (binEdge1[:-1] + binEdge1[1:])/2.0

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 3.9))
        ax1.plot(xres, ADres1, 'r--', label=f'AD limit')
        ax1.plot(binCenter1, binVal1, 'g', label=f'MC')
        ax1.set_xlim((-5, 5))
        ax1.legend()
        ax1.set_title(f'Advection-Diffusion limit test with scattering rate: {scatteringRate1}')

        fig.savefig('MPIExample - AD test')

    t2 = time.perf_counter()
    print(f'Process {myrank} shutting down. Was alive for {t2-t1} seconds.')