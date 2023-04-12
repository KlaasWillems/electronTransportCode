from typing import Literal, TypeAlias, Final, Annotated
import numpy.typing as npt
import numpy as np
from scipy import constants

# predefined types
tuple2d: TypeAlias = Annotated[npt.NDArray[np.float64], Literal[2]]
tuple3d = Annotated[npt.NDArray[np.float64], Literal[3]]

# Constants
CTF: Final[float] = np.power(9*(np.pi**2)/128, 1/3)  # Thomas-Fermi constant. Dimensionless
FSC: Final[float] = constants.fine_structure  # fine structure constant (denoted with symbol alpha). Dimensionless
Z_WATER: Final[float] = 10  # atomic number of water
Z_WATER_EFF: Final[float] = 7.51  # effective atomic number of water
ERE: Final[float] = constants.value('electron mass energy equivalent in MeV')  # [MeV] electron rest energy in MeV
A_WATER: Final[float] = 18  # Relative molar mass of water
I_WATER: Final[float] = 75  # [eV] mean excitation energy for water
Re: Final[float] = constants.value('classical electron radius')*100  # [cm] classical electron radius
SC_DENSITY_WATER: Final[float] = 3.3428847*1e22  # [cm^-3] scattering center density for water. Electron density of water divided by Z_WATER. See Olbrant et al.
E_THRESHOLD: Final[float] = min(0.0003162278/ERE, I_WATER/(1e6*ERE))  # Stopping power becomes negative if energy is lower than this value
RHO_WATER: Final[float] = 1.0  # [g/cm^3] density of water
