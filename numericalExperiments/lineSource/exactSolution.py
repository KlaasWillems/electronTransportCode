"""Exact solution to line source benchmarkt for sigma = 1.0. See Garret & Hauck 2013
"""

import numpy as np
from scipy.integrate import quad

HEAVISIDE_THRESHOLD = 1.0

def lineSourceSolution(x: float, y: float, E: float, Emax: float) -> float:
    t = Emax - E
    R = np.sqrt(np.power(x, 2) + np.power(y, 2))
    gamma = R/t

    term1 = np.exp(-t)/(2*np.pi*np.power(t, 2)*np.sqrt(1-np.power(gamma, 2)))
    integrand = lambda omega: ptc(t*np.sqrt(np.power(gamma, 2) + np.power(omega, 2)), t)

    i1, _ = quad(integrand, 0, np.sqrt(1-np.power(gamma, 2)))

    return (term1 + 2*t*i1)*np.heaviside(1-gamma, HEAVISIDE_THRESHOLD)

def ptc(R: float, t: float) -> float:
    return pt1(R, t) + pt2(R, t)

def beta(u: float, q: float, gamma: float) -> float:
    return (np.log(q) + 1j*u)/(gamma + 1j*np.tan(u/2))

def pt1(R: float, t: float) -> float:
    gamma = R/t
    return np.exp(-t)*np.log((1.0 + gamma)/(1.0 - gamma))/(4*np.pi*R*t)

def pt2(R: float, t: float) -> float:
    gamma = R/t
    q = (1.0+gamma)/(1.0-gamma)
    const1 = np.exp(-t)*(1 - np.power(gamma, 2))*np.heaviside(1-gamma, HEAVISIDE_THRESHOLD)/(32*np.power(np.pi, 2)*R)

    integrand = lambda u: (np.power(np.tan(u/2), 2)+1.0)*ReFunc(u, q, gamma, t)
    i1, _ = quad(integrand, 0, np.pi)

    return const1*i1

def ReFunc(u: float, q: float, gamma: float, t: float) -> float:

    b = beta(u, q, gamma)
    term1 = (gamma + 1j*np.tan(u/2))*np.power(b, 3)
    term2 = np.exp(b*t*(1-np.power(gamma, 2))/2)

    return np.real(term1*term2)
