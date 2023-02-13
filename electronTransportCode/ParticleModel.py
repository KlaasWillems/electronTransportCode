from abc import ABC, abstractmethod
import numpy as np
from scipy import constants
from typing import Tuple, Final


CTF: Final[float] = np.power(9*(np.pi**2)/128, 1/3)  # Thomas-Fermi constant
FSC: Final[float] = constants.value('fine_structure')  # fine structure constant (denoted with symbol alpha)
Z_WATER: Final[float] = 10  # atomic number of water
Z_WATER_EFF: Final[float] = 7.51  # effective atomic number of water 
ERE: Final[float] = constants.value('electron mass energy equivalent in MeV')  # [MeV] electron rest energy in MeV
A_WATER: Final[float] = 18  # Relative molar mass of water
I_WATER: Final[float] = 75  # [eV] mean excitation energy for water
Re: Final[float] = constants.value('classical electron radius')*100  # [cm] classical electron radius
NB_DENSITY_WATER: Final[float] = 3.3428847  # [cm^-3] electron number density of water
E_THRESHOLD: Final[float] = 1.0

class ParticleModel(ABC):
    
    @abstractmethod
    def sampleCrossSection(self, Ekin: float, rho: float, Z: float = Z_WATER, A: float = A_WATER) -> Tuple[float, float]:
        """Sample path-length and polar scattering angle. 

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            rho (float): mass density of medium
            Z (float): atomic number. Defaults to Z_WATER.
            A (float): relative molecular mass. Defaults to A_WATER.

        Returns:
            float: path-length [cm] and polar scattering angle
        """

        pass
            
    @abstractmethod
    def evalStoppingPower(self, Ekin: float, Ec: float = E_THRESHOLD, I: float = I_WATER, NB_DENSITY: float = NB_DENSITY_WATER) -> float:
        """Evaluate electron stopping power. 

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            DeltaE (float): Energy cut-off value for soft-inelastic collisions in the same units as Ekin. Defaults to E_THRESHOLD.
            I (float, optional): Mean excitation energy. Defaults to I_WATER.
            NB_DENSITY (float): Number density of scattering medium. Defaults to NB_DENSITY_WATER.

        Returns:
            float: Stopping power evaluated at Eking and DeltaE [MeV/cm]
        """
        pass
    
    
class SimplifiedEGSnrcElectron(ParticleModel):
    
    def sampleCrossSection(self, Ekin: float, rho: float, Z: float = Z_WATER, A: float = A_WATER) -> Tuple[float, float]:
        """ Sample path-length and polar scattering angle from screened Rutherford elastic scattering cross section. See EGSnrc manual by Kawrakow et al for full details.
            
            See abstract base class method for arguments and return value.
        """
        # Polar scattering angle
        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1,2)
        beta: float = np.sqrt(betaSquared)
        alfaPrime: float = FSC*Z/beta
        eta0: float = np.power(FSC, 2)*np.power(Z, 2/3)/(4*np.power(CTF, 2)*Ekin*(Ekin+2))
        r: float = np.random.uniform()
        eta: float = eta0*(1.13 + 3.76*alfaPrime**2)
        mu: float = 1 - 2*eta*r/(1-r+eta)  # polar scattering angle mu
        
        # path-length
        ZS: float = Z*(Z + 1)
        ZE: float = Z*(Z + 1)*np.log(np.power(Z, -2/3))
        ZX: float = Z*(Z + 1)*np.log(1 + 3.34*np.power(FSC*Z, 2))
        bc: float = 7821.6 * rho * ZS * np.exp(ZE/ZS)/(A * np.exp(ZX/ZS))
        SigmaSR: float = bc/betaSquared  # total macroscopic screened Rutherford cross section
        s: float = np.random.exponential(1/SigmaSR)  # path-length 
        return s, mu

    def evalStoppingPower(self, Ekin: float, Ec: float = E_THRESHOLD, I: float = I_WATER, NB_DENSITY: float = NB_DENSITY_WATER) -> float:
        """ Restricted stopping power from EGS4 and EGSnrc based on Bethe-Bloch theory. Implementation does not include density effect correction that takes into account 
            the polarization of the medium due to the electron field.

            See abstract base class method for arguments and return value.
        """

        Ekin_eV: float = Ekin*ERE*1e6  # Electron kinetic energy in eV (E or T in literature)
        delta: float = 0.0
        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1, 2)
        eta: float = Ec/Ekin
        G: float = -1 - betaSquared + np.log(4*eta*(1-eta)) + 1/(1-eta) + (1 - betaSquared)*(np.power(Ekin*eta, 2)/2 + (2*Ekin + 1)*np.log(1-eta))
        Lcoll: float = 2*np.pi*np.power(Re, 2)*ERE*NB_DENSITY*(2*np.log(Ekin_eV/I) + np.log(1 + Ekin/2) + G - delta)/betaSquared
        return Lcoll