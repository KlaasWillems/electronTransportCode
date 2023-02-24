import numpy as np
from electronTransportCode.ProjectUtils import tuple2d
from scipy.integrate import quad

HEAVISIDE_THRESHOLD = 1.0

def lineSourceSolution(sigma: float, x: float, y: float, E: float, Emax: float) -> float:
    x = sigma*x
    y = sigma*y
    t = sigma*(Emax - E)
    R = np.sqrt(np.power(x, 2) + np.power(y, 2))
    gamma = R/t
    q = (1.0 + gamma)/(1.0 - gamma)

    term1 = np.exp(-t)/(2*np.pi*np.power(t, 2)*np.sqrt(1-np.power(gamma, 2)))
    integrand = lambda omega: ptc(t*np.sqrt(np.power(gamma, 2) + np.power(omega, 2)), t, gamma, q)

    i1, _ = quad(integrand, 0, np.sqrt(1-np.power(gamma, 2)))

    return (term1 + 2*t*i1)*np.heaviside(1-gamma, HEAVISIDE_THRESHOLD)

def ptc(R: float, t: float, gamma: float, q: float) -> float:
    return pt1(R, t, gamma) + pt2(R, t, gamma, q)

def beta(u: float, q: float, gamma: float) -> float:
    return (np.log(q) + 1j*u)/(gamma + 1j*np.tan(u/2))

def pt1(R: float, t: float, gamma: float) -> float:
    return np.exp(-t)*np.log((1.0 + gamma)/(1.0 - gamma))/(4*np.pi*R*t)

def pt2(R: float, t: float, gamma: float, q: float) -> float:
    const1 = np.exp(-t)*(1 - np.power(gamma, 2))*np.heaviside(1-gamma, HEAVISIDE_THRESHOLD)/(32*np.power(np.pi, 2)*R)

    integrand = lambda u: (np.power(np.tan(u/2), 2)+1.0)*ReFunc(u, q, gamma, t)
    i1, _ = quad(integrand, 0, np.pi)

    return const1*i1

def ReFunc(u: float, q: float, gamma: float, t: float) -> float:

    b = beta(u, q, gamma)
    term1 = (gamma + 1j*np.tan(u/2))*np.power(b, 3)
    term2 = np.exp(b*t*(1-np.power(gamma, 2))/2)

    return np.real(term1*term2)
