from typing import TypeAlias, Final
import numpy.typing as npt
import numpy as np
from scipy import constants

# predefined types
tuple2d: TypeAlias = npt.NDArray[np.float64]

# Constants
CTF: Final[float] = np.power(9*(np.pi**2)/128, 1/3)  # Thomas-Fermi constant. Dimensionless
FSC: Final[float] = constants.fine_structure  # fine structure constant (denoted with symbol alpha). Dimensionless
Z_WATER: Final[float] = 10  # atomic number of water
Z_WATER_EFF: Final[float] = 7.51  # effective atomic number of water
ERE: Final[float] = constants.value('electron mass energy equivalent in MeV')  # [MeV] electron rest energy in MeV
A_WATER: Final[float] = 18  # Relative molar mass of water
I_WATER: Final[float] = 75  # [eV] mean excitation energy for water
Re: Final[float] = constants.value('classical electron radius')*100  # [cm] classical electron radius
NB_DENSITY_WATER: Final[float] = 3.3428847*1e23  # [cm^-3] electron number density of water
E_THRESHOLD: Final[float] = 0.5
RHO_WATER: Final[float] = 1.0  # [g/cm^3] density of water
