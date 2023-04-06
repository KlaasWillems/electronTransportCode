import numpy as np
from typing import Tuple
from electronTransportCode.ProjectUtils import tuple3d

sp = '0.2*(1 + E**2)'
dS = 2.50
minEnergy = 0.0

def evalStoppingPower(Ekin: float, pos3d: tuple3d) -> float:
    if isinstance(sp, float) or isinstance(sp, int):
        return sp
    else:
        if sp == '(1 + x**2)':
            return 1 + pos3d[0]**2
        elif sp == '(1 + 0.05*cos(x*2*3.1415/20))':
            return 1 + 0.05*np.cos(pos3d[0]*2*3.1415/20)
        elif sp == '(1 + 0.5*sin(x))':
            return 1.0 + 0.5*np.sin(pos3d[0])
        elif sp == '0.2*(1 + E**2)':
            return (1.0 + Ekin**2)*0.2
        elif sp == 'E + 10':
            return Ekin + 10.0
        else:
            raise NotImplementedError('Invalid stopping power')

def energyLoss(Ekin: float, pos3d: tuple3d, stepsize: float) -> float:
    assert Ekin > 0, f'{Ekin=}'
    assert stepsize > 0, f'{stepsize=}'
    Emid = Ekin + evalStoppingPower(Ekin, pos3d)*stepsize/2
    assert Emid > 0, f'{Emid=}'
    return evalStoppingPower(Emid, pos3d)*stepsize

def pickStepSize(kin_pos3d, kin_energy, step_kin, N) -> Tuple[float, float]:
    step_diff = dS - (step_kin % dS)
    deltaE = energyLoss(kin_energy, kin_pos3d, step_diff)
    new_energy = kin_energy - deltaE

    if new_energy < minEnergy:
        Erange = np.linspace(kin_energy, minEnergy, N)  # energy from minEenergy to kin_energy
        middle = [(Erange[i] + Erange[i+1])/2 for i in range(N-1)]  # energy at the middle of the intervals
        sps = [evalStoppingPower(e, kin_pos3d) for e in middle]  # stopping power at middle of the intervals
        dss = -np.diff(Erange)/sps  #
        return sum(dss), kin_energy-minEnergy

    return -1, -1

def pickStepSize2(kin_pos3d, kin_energy, step_kin, N) -> Tuple[float, float]:
    step_diff = dS - (step_kin % dS)
    deltaE = energyLoss(kin_energy, kin_pos3d, step_diff)
    new_energy = kin_energy - deltaE

    if new_energy < minEnergy:
        Erange = np.linspace(kin_energy, minEnergy, N)  # energy from minEenergy to kin_energy
        sps = [evalStoppingPower(e, kin_pos3d) for e in Erange]
        dss = [2*(Erange[i] - Erange[i+1])/(sps[i] + sps[i+1]) for i in range(N-1)]
        return sum(dss), kin_energy-minEnergy

    return -1, -1


pos = np.array((1.0, 0.0, 0.0), dtype=float)
kin_energy = 0.6
step_kin = 0.25

print(pickStepSize(pos, kin_energy, step_kin, N=5))
print(pickStepSize(pos, kin_energy, step_kin, N=50))
print(pickStepSize(pos, kin_energy, step_kin, N=500))

print(pickStepSize2(pos, kin_energy, step_kin, N=5))
print(pickStepSize2(pos, kin_energy, step_kin, N=50))
print(pickStepSize2(pos, kin_energy, step_kin, N=500))