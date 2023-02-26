"""Exact solution to point source benchmarkt for sigma = 1.0. See Garret & Hauck 2013 and Ganapol 1999
"""
import numpy as np
from scipy.integrate import quad

HEAVISIDE_THRESHOLD = 1.0
C: float = 1.0/(2.0*np.pi)  # c constant in Ganapol 1999

def pointSourceSolution(x: float, y: float, E: float, Emax: float) -> float:
    """Exact solution to radition equation with a pulsed point isotropic source assuming homogenous material, constant unit scattering rate, unit density and unit stopping power.

    Args:
        x (float): x coordinate
        y (float): y coordinate
        E (float): Energy of particle
        Emax (float): Initial energy of particle

    Returns:
        float: Density at coordinate
    """
    t = Emax - E
    R = np.sqrt(np.power(x, 2) + np.power(y, 2))
    gamma = R/t
    if gamma == 1.0:
        return pt0(R, t)*2
    else:
        return pt1(R, t) + ptplus(R, t)

def pt0(R: float, t: float) -> float:
    """Density of particles that haven't collided. See equation 13 Ganapol 1999
    """
    gamma = R/t
    if gamma == 1.0:  # Dirac delta at gamma == 1.0
        return np.exp(-t)/(4*np.pi*R*np.power(t, 2))
    else:
        return 0.0

def pt1(R: float, t: float) -> float:
    """Density of particles that have collided once. See equation 13 Ganapol 1999
    """
    gamma = R/t
    term1 = np.exp(-t)*np.log((1.0 + gamma)/(1.0 - gamma))*C/(4*np.pi*R*t)
    return term1

def ptplus(R: float, t: float) -> float:
    """Density of particles that have collided at least two times. See equation 13 Ganapol 1999
    """
    gamma = R/t
    q = (1.0 + gamma)/(1.0 - gamma)
    integrand = lambda u: (np.power(np.tan(u/2), 2) + 1.0)*ReFunc(u, q, gamma, t)

    i1, _ = quad(integrand, 0.0, np.pi)
    return np.exp(-t)*np.power(C, 2)*(1 - np.power(gamma, 2))*i1*np.heaviside(1-gamma, HEAVISIDE_THRESHOLD)/(np.power(np.pi, 2)*32*R)

def ReFunc(u: float, q: float, gamma: float, t: float) -> float:
    """Real part in equation 13 Ganapol 1999
    """
    b = beta(u, q, gamma)
    term1 = (gamma + 1j*np.tan(u/2))*np.power(b, 3)
    term2 = np.exp(b*t*C*(1-np.power(gamma, 2))/2)

    return np.real(term1*term2)

def beta(u: float, q: float, gamma: float) -> float:
    """Xi function in equation 13 Ganapol 1999
    """
    return (np.log(q) + 1j*u)/(gamma + 1j*np.tan(u/2))
