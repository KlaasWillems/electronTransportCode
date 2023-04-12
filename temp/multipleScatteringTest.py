import math
from typing import Final
from electronTransportCode.Material import Material, WaterMaterial
from electronTransportCode.ProjectUtils import ERE, tuple3d, FSC
from electronTransportCode.ParticleModel import SimplifiedEGSnrcElectron

RNGSEED: Final[int] = 4

particle = SimplifiedEGSnrcElectron(RNGSEED)
material = WaterMaterial

def sampleMS(Ekin: float, EkinOld: float, pos: tuple3d, material: Material, stepsize: float = 0.1) -> float:
    """Sample multiple-scattering distribution for an electron for path-length 'stepsize', which corresponds to an energy loss E0 - E.

    Args:
        Ekin (float): Next energy (smallest) in units of ERE
        Ekin0 (float): Old energy in units of ERE
        material (Material): Material of cell
        stepsize (float, optional): KD Stepsize. Defaults to 0.1.

    Returns:
        float: Sample of scattering angle mu
    """
    E = Ekin*ERE  # Energy in MeV

    eps = (EkinOld - Ekin)/Ekin
    tau_t = (EkinOld + Ekin)/2  # Units of ERE
    num1 = 2.0*(tau_t**2) + 10.0*tau_t + 6.0
    denum1 = (tau_t+1.0)*(tau_t+2.0)
    betaSquared = Ekin*(Ekin+2)/((Ekin+1)**2)
    beta = math.sqrt(betaSquared)

    # Finite difference approximation of derivative
    delta = 1e-8
    SEmid = particle.evalStoppingPower(Ekin, pos, material)*ERE
    SEplus = particle.evalStoppingPower(Ekin+delta, pos, material)*ERE
    SEmin = particle.evalStoppingPower(Ekin-delta, pos, material)*ERE
    dS = (SEplus-SEmin)/(2*delta)

    # Compute Eff (4.8.15)
    CE = SEmid*betaSquared
    dbetaE = 2/(E**3 + 3*E**2 + 3*E + 1.0)
    dCE = dS*betaSquared + 2*beta*dbetaE*SEmid
    b_term = E*dCE/CE
    Eeff = EkinOld*(1.0 - eps/2 - ((eps**2)/(12*(2.0-eps)))*(num1/denum1 + 2*b_term)  )*ERE  # in MeV

    # Compute effective stepsize (4.8.16)
    num2 = tau_t**4 + 4*tau_t**3 + 7*tau_t**2 + 6*tau_t + 4
    denum2 = ((tau_t+1.0)**2)*((tau_t+2.0)**2)
    seff = stepsize*(1.0 - (eps**2)*num2/(denum2*3*(2.0-eps)) )

    # Compute lambda and eta
    SigmaSR = particle.getScatteringRate(pos, Ekin, material)
    l = SigmaSR*seff  # lambda
    t = math.log(l)
    alfaPrime: float = FSC*material.Z/beta
    eta0: float = material.eta0CONST/(Ekin*(Ekin+2))
    eta: float = eta0*(1.13 + 3.76*(alfaPrime**2))

    assert particle.rng is not None
    r1 = particle.rng.uniform()
    r2 = particle.rng.uniform()

    if r1 < math.exp(-l):
        return 0.0
    elif r1 < math.exp(-l)*(1 + l):  # Single collision occured
        return 1 - 2*eta*r2/(1-r2+eta)
    else:  # Multiple collisions occured
        G1 = 2*l*eta*((1+eta)*math.log(1+1/eta)-1)
        if l < 10:
            wSquared = (1.347 + t*(0.209364 - t*(0.45525 - t*(0.50142 - 0.081234*t))))*(l+4)
        else:
            wSquared = -2.77164 + t*(2.94874 - t*(0.1535754 - 0.00552888*t))
        a = wSquared*eta

        # Lookup u
        # https://github.com/nrc-cnrc/EGSnrc/tree/master/HEN_HOUSE/data
        # msnew.data
        # See Kawrakow et al. secion on MS without spin correction

        return 2*a*u/(1-u+a)
