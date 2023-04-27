import math
from typing import Final, Tuple
from itertools import islice
import numpy as np
from electronTransportCode.ProjectUtils import FSC, ERE, mathlog2
import numba as nb

MAXL_MS: Final[int] = 63
MAXQ_MS: Final[int] = 7
MAXU_MS: Final[int] = 31
LAMBMIN_MS: Final[float] = 1.0
LAMBMAX_MS: Final[float] = 1e6
QMIN_MS: Final[float] = 1e-3
QMAX_MS: Final[float] = 0.5

def loadTable() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float]:
    ums_array = np.empty(shape=(MAXL_MS+1, MAXQ_MS+1, MAXU_MS+1), dtype=float)
    fms_array = np.empty(shape=(MAXL_MS+1, MAXQ_MS+1, MAXU_MS+1), dtype=float)
    wms_array = np.empty(shape=(MAXL_MS+1, MAXQ_MS+1, MAXU_MS+1), dtype=float)
    ims_array = np.empty(shape=(MAXL_MS+1, MAXQ_MS+1, MAXU_MS+1), dtype=int)
    with open('data/msnew.data', 'r') as reader:
        for i in range(MAXL_MS+1):
            for j in range(MAXQ_MS+1):
                line_gen = list(islice(reader, 14))
                k = 0
                for index in range(0, 4):  # part 1
                    line = line_gen[index]
                    lineSplit = line[2:-1].split('  ')
                    for x in lineSplit:
                        ums_array[i, j, k] = float(x)
                        k += 1
                k = 0
                for index in range(4, 12):  # part 2
                    line = line_gen[index]
                    lineSplit = line[1:-1].split(' ')
                    for x in lineSplit:
                        if k < 32:
                            fms_array[i, j, k] = float(x)
                        else:
                            wms_array[i, j, k-32] = float(x)
                        k += 1
                k = 0
                for index in range(12, 14):  # part 3
                    line = line_gen[index]
                    formattedLine = line[:-1].split(' ')
                    lineSplit = [x for x in formattedLine if x != '']
                    for x in lineSplit:
                        ims_array[i, j, k] = int(x)
                        k += 1
                # Do calculation for ims & fms
                for k in range(MAXU_MS):
                    fms_array[i, j, k] = fms_array[i,j,k+1]/fms_array[i, j, k] - 1
                    ims_array[i, j, k] = ims_array[i, j, k] - 1
                fms_array[i, j, MAXU_MS] = fms_array[i, j, MAXU_MS-1]
    llammin = math.log(LAMBMIN_MS); llammax = math.log(LAMBMAX_MS)
    dllamb = (llammax-llammin)/MAXL_MS; dllambi = 1/dllamb
    dqms = QMAX_MS/MAXQ_MS; dqmsi = 1/dqms
    return ums_array, fms_array, wms_array, ims_array, llammin, llammax, dllamb, dllambi, dqms, dqmsi

ums_array, fms_array, wms_array, ims_array, llammin, llammax, dllamb, dllambi, dqms, dqmsi = loadTable()

@nb.jit(nb.float64(nb.float64, nb.float64, nb.float64), nopython=True, cache=True)
def evalStoppingPower(Ekin: float, materialI: float, materialLcoll: float) -> float:
    """ Stopping power from PENELOPE for close and distant interactions. Equation 3.120 in PENELOPE 2018 Conference precedings.
        - This previous implementation did not include density effect correction that takes into account the polarization of the medium due to the electron field.
    """
    # Ekin = tau in papers
    Ekin_eV: float = Ekin*ERE*1e6  # Electron kinetic energy in eV (E or T in literature)
    # assert Ekin_eV >= materialI, f'Input energy: {Ekin_eV} is lower than I'

    betaSquared: float = Ekin*(Ekin+2)/((Ekin+1)**2)

    gamma = Ekin+1
    argument = ((Ekin_eV/materialI)**2)*((gamma+1)/2)
    if argument > 1e-320:  # Roundings errors in argument can cause ValueError in math.log
        term1 = math.log(argument)
    else:
        term1 = math.log(1e-320)
    term2 = 1 - betaSquared - ((2*gamma - 1)/(gamma**2))*mathlog2 + (((gamma-1)/gamma)**2)/8
    Lcoll = materialLcoll*(term1 + term2)/betaSquared

    return Lcoll

@nb.jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True, cache=True)
def energyLoss(Ekin: float, stepsize: float, materialI: float, materialLcoll: float) -> float:
    """Compute energy along step using continuous slowing down approximation. Eq. 4.11.3 from EGSnrc manual, also used in GPUMCD.
    Approximation is second order accurate: O(DeltaE^2)

    Args:
        Ekin (float): Energy at the beginning of the step. Energy unit relative to electron rest energy.
        pos3d (tuple3d): position of particle
        stepsize (float): [cm] path-length
        material (Material): Material of current grid cell

    Returns:
        float: Energy loss DeltaE
    """
    # assert Ekin*ERE*1e6 >= materialI, f'{Ekin=}'
    # assert stepsize > 0, f'{stepsize=}'
    Emid = Ekin + evalStoppingPower(Ekin, materialI, materialLcoll)*stepsize/2
    # assert Emid*ERE*1e6 >= materialI, f'{Emid=}'
    return evalStoppingPower(Emid, materialI, materialLcoll)*stepsize

@nb.jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True, cache=True)
def mscat(Ekin: float, stepsize: float, EkinLoss: float, Z: float, eta0CONST: float, bc: float) -> float:

    temporary = Ekin*(Ekin+2)
    betaSquared = temporary/((Ekin+1)**2)
    eta0 = eta0CONST/temporary
    chia2 = eta0*(1.13 + 3.76*((FSC*Z)**2)/betaSquared)

    lambdavar  = stepsize*bc/betaSquared
    epsilonp = EkinLoss/Ekin
    temp2  = 0.166666*(4+Ekin*(6+Ekin*(7+Ekin*(4+Ekin))))*(epsilonp/((Ekin+1)*(Ekin+2)))**2;
    lambdavar = lambdavar*(1 - temp2)/(1 + chia2)

    chilog = math.log(1 + 1/chia2)
    q1 = 2*chia2*(chilog*(1 + chia2) - 1)*lambdavar

    # chia2 (float): screening angle
    # q1 (float): first moment of the single scattering cross section
    # lambda (float): distance in number of elastic scattering mean free paths

    k: int; i: int; j: int

    if lambdavar <= 13.8:
        sprob = np.random.uniform(0.0, 1.0)
        explambda = math.exp(-lambdavar)
        if sprob < explambda:
            # It was a no scattering event
            cost = 1.0
            return cost
        wsum = (1+lambdavar)*explambda
        if sprob < wsum:
            xi = np.random.uniform(0.0, 1.0)
            xi  = 2*chia2*xi/(1 - xi + chia2)
            cost = 1.0 - xi
            return cost
        if lambdavar <= 1:
            wprob = explambda; wsum = explambda
            cost = 1.0; sint = 0.0
            icount = 0; loopBool = True
            while loopBool:
                icount = icount + 1
                if icount > 20:
                    break
                wprob = wprob*lambdavar/icount
                wsum = wsum + wprob
                xi = np.random.uniform(0.0, 1.0)
                xi  = 2*chia2*xi/(1 - xi + chia2)
                cosz = 1 - xi
                sinz = xi*(2 - xi)
                if sinz > 1.e-20:
                    sinz = math.sqrt(sinz)
                    xi = np.random.uniform(0.0, 1.0)
                    phi = xi*6.2831853
                    cost = cost*cosz - sint*sinz*math.cos(phi)
                    sint = math.sqrt(max(0.0,(1-cost)*(1+cost)))
                if wsum > sprob:
                    loopBool = True
                else:
                    loopBool = False
            return cost

    # It was a multiple scattering event
    # Sample the angle from the q^(2+) surface
    if lambdavar <= LAMBMAX_MS:

        llmbda = math.log(lambdavar)

        ai = llmbda*dllambi; i = int(ai); ai = ai - i
        xi = np.random.uniform(0.0, 1.0)
        if xi < ai:
            i = i + 1

        if q1 < QMIN_MS:
            j = 0

        elif q1 < QMAX_MS:
            aj = q1*dqmsi; j = int(aj); aj = aj - j
            xi = np.random.uniform(0.0, 1.0)
            if xi < aj:
                j = j + 1
        else:
            j = MAXQ_MS

        if llmbda < 2.2299:
            omega2 = chia2*(lambdavar + 4)*(1.347006 + llmbda*(0.209364 - llmbda*(0.45525 - llmbda*(0.50142 - 0.081234*llmbda))))
        else:
            omega2 = chia2*(lambdavar + 4)*(-2.77164 + llmbda*(2.94874 - llmbda*(0.1535754 - llmbda*0.00552888)))

        xi = np.random.uniform(0.0, 1.0)
        ak = xi*MAXU_MS; k = int(ak); ak = ak - k
        if ak > wms_array[i,j,k]:
            k = ims_array[i,j,k]
        a = fms_array[i,j,k]; u = ums_array[i,j,k]
        du = ums_array[i,j,k+1] - u
        xi = np.random.uniform(0.0, 1.0)
        if abs(a) < 0.2:
            x1 = 0.5*(1-xi)*a
            u  = u + xi*du*(1+x1*(1-xi*a))
        else:
            u = u - du/a*(1-math.sqrt(1+xi*a*(2+a)))

        xi = omega2*u/(1 + 0.5*omega2 - u)
        if xi > 1.99999:
            xi = 1.99999
        cost = 1 - xi
        return cost

    # raise ValueError(f'lambda: {lambdavar}')
    raise ValueError(f'lambda too large')
