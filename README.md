# Introduction
This repository contains the code that was developed during my master thesis at the KU Leuven on 'Particle, Fluid and Hybrid Numerical Methods for Radiation Therapy' in 2022-2023.   

# Dependencies
A full state of all the pip packages that I had installed in the virtual environment is given in the requirements.txt file (contains a lot of unused packages).

## Dependencies for running the code
- Python
- Scipy
- Numpy
- mpi4py
- Numba
- [spherical_stats](https://github.com/dschmitz89/spherical_stats) 

## Dependencies for running the jupyter notebooks
- pingouin (for some Q-Q plots)
- pillow (for working with lung image)
- py-pde (for solving advection-diffusion equation)
- SymPy
- Pandas
- Matplotlib

# Repository overview
The source code can be found in the electronTransportCode/ folder in the top directory of this repository. All numerical experiments that are documented in the thesis are situated in the numericalExperiments/ or ms/ folders. The tests/ directory contains some unittests for the SimulationDomain and MCEstimtor objects.

## Source code
At its core, the particle transport code developed for this thesis consists of five Python objects. The abstract _MCEstimator_ object defines all operations relevant to scoring a quantity of interest. Each quantity of interest (e.g. fluence, dose, ...) is implemented in a separate class which inherits from _MCEstimator_. The main method, `updateEstimator', takes in the old and new state of a particle, and scores the quantity of interest after each step. When a simulation is parallelized using MPI, each process independently scores quantities of interest and combines them at the end of the simulation. 

The _SimulationDomain_ object encapsulates all methods relevant to the positioning of a particle. A simulation domain is three-dimensional. In the x-dimension, the domain is infinite. In other words, there is no discretisation and boundary checking in the x-direction. The y and z direction are discretised on a two-dimensional rectangular grid (in other words, the domain is discretized in rectangular cuboids). Each grid cell is assigned an identification number or index (indices are row-wise ordered in the two-dimensional y-z grid). Throughout the simulation, the code keeps track of the particle's position and the index of the grid cell in which the particle is moving. Based on the index, the code knows the material properties of the cell. Lastly, the object can also compute the distance of a particle to the nearest boundary. This is used to check for geometric events. 

The _ParticleModel_ object defines a particle's scattering characteristics. It implements the sampling routine for the step size and scattering angle. In addition, the object uses the approximation from GPUMCD and EGSnrc to compute the energy loss of a particle along a step. 

The _SimOptions_ object implements the initial condition of a simulation (position, energy and velocity). In addition, it stores the random number generator object and several simulation parameters like the threshold energy $E_{th}$.

Finally, the _ParticleTracer_ object implements several particle tracing algorithms: analog particle tracing algorithm, kinetic-diffusion Monte Carlo and kinetic-diffusion-rotation Monte Carlo. In case many particles need to be simulated, the simulation can be parallelized using mpi4py. MPI was chosen over Python's multiprocessing library since MPI scales better on distributed memory processors. In addition, since the particles do not interact, there is no need for any communication during the simulation. 

## Scripts & experiments
Here, an overview is given of the scripts that were used to generate the results in the thesis.

- ms
    - averageMU: plot the screened Rutherford elastic differential cross-section and some of its moments. 
    - msAngleTest: Generate multiple scattering distribution for the velocity after a diffusive step.
    - ownMS: Generate lookup table for the variance of kinetic motion for KDR.
    - velocitDist: Fit several different analytical distributions through multiple scattering velocity distribution and test goodness-of-fit.
- numericalExperiments
    - KDMCTest
        - AdvectionTest: Try KDMC with space-dependent scattering rate.
        - momentumQOI: Showcase KDMC in case the quantity of interest requires scoring of the velocity.
        - pointSourceDS: Error of KDMC as a function of step size.
        - pointSourceR: Error of KDMC as a function of scattering rate.
        - simpleHomogeneous: Test KDMC in simplified setting with homogeneous scattering parameters.
        - speedUpTest: Speed-graph of KDMC.
    - KDRTest:
        - kdrTest: Estimate particle distribution with KDR.
        - speedUpTest: Speed-graph of KDR.
        - KuschLungTest: Do dose estimation on 2D lung test case from Prof. J. Kusch. See his [repository](https://github.com/JonasKu/publication-A-robust-collision-source-method-for-rank-adaptive-dynamical-low-rankapproximation).
    - diffusionLimit: Several tests to check advection-diffusion limit derived for KDMC.
    - lineSource: Analytical line source benchmark for analog particle tracing algorithm.
    - singleScatteringCorrelation: Plot angle between velocities along a single track of a particle.
    - waterPhantom: Waterphantom test case for analog particle tracing algorithm.


