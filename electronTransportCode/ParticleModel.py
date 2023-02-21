from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
from electronTransportCode.Material import Material
from electronTransportCode.ProjectUtils import ERE, FSC, Re


class ParticleModel(ABC):
    """A particle is defined by its path-length distribution, angular scattering distribution and stopping power.
    """
    def __init__(self, generator: Union[np.random.Generator, None, int] = None) -> None:
        self.rng: Optional[np.random.Generator]
        if isinstance(generator, int):
            self.rng = np.random.default_rng(generator)
        elif generator is None:
            self.rng = None
        elif isinstance(generator, np.random.Generator):
            self.rng = generator
        else:
            raise ValueError('Generator input argument invalid.')

    def setGenerator(self, generator: np.random.Generator) -> None:
        """Store random number generator object as class member

        Args:
            generator (np.random.Generator):
        """
        self.rng = generator

    @abstractmethod
    def samplePathlength(self, Ekin: float, material: Material) -> float:
        """Sample path-length.

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            material (Material): material of scattering medium in cell.

        Returns:
            float: path-length [cm]
        """

    @abstractmethod
    def sampleAngle(self, Ekin: float, material: Material) -> float:
        """Sample polar scattering angle mu.

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            material (Material): material of scattering medium in cell.

        Returns:
            float: polar scattering angle
        """

    @abstractmethod
    def evalStoppingPower(self, Ekin: float, material: Material) -> float:
        """Evaluate electron stopping power.

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            material (Material): material of scattering medium in cell.
            DeltaE (float): Energy cut-off value for soft-inelastic collisions in the same units as Ekin. Defaults to E_THRESHOLD.

        Returns:
            float: Stopping power evaluated at Ekin and DeltaE [1/cm] (energy relative to electron rest energy)
        """


class LineSourceParticle(ParticleModel):
    """Particle for line source benchmark. See Kush & Stammer paper.
    """
    def __init__(self, generator: Union[np.random.Generator, None, int] = None) -> None:
        super().__init__(generator)
        self.sigma = 1

    def samplePathlength(self, Ekin: float, material: Material) -> float:
        assert self.rng is not None
        return self.rng.exponential(scale=1)/self.sigma

    def sampleAngle(self, Ekin: float, material: Material) -> float:
        # Isotropic scattering angle
        assert self.rng is not None
        return self.rng.uniform(low=-1, high=1)

    def evalStoppingPower(self, Ekin: float, material: Material) -> float:
        return 1


class SimplifiedEGSnrcElectron(ParticleModel):
    """A simplified electron model. Soft elastic collisions are taken into account using the screened Rutherford elastic cross section.
    Energy loss is deposited continuously using the Bethe-Bloch inelastic restricted collisional stopping power. Hard-inelastic collisions and bremstrahlung are not
    taken into account.
    """

    def samplePathlength(self, Ekin: float, material: Material) -> float:
        """ Sample path-length from screened Rutherford elastic scattering cross section. See EGSnrc manual by Kawrakow et al for full details.

            See abstract base class method for arguments and return value.
        """
        assert self.rng is not None

        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1,2)
        SigmaSR: float = material.bc/betaSquared  # total macroscopic screened Rutherford cross section
        return self.rng.exponential(1/SigmaSR)  # path-length

    def sampleAngle(self, Ekin: float, material: Material) -> float:
        """ Sample polar scattering angle from screened Rutherford elastic scattering cross section. See EGSnrc manual by Kawrakow et al for full details.

            See abstract base class method for arguments and return value.
        """
        assert self.rng is not None
        Z = material.Z

        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1,2)
        beta: float = np.sqrt(betaSquared)
        alfaPrime: float = FSC*Z/beta
        eta0: float = material.eta0CONST/(Ekin*(Ekin+2))
        r: float = self.rng.uniform()
        eta: float = eta0*(1.13 + 3.76*alfaPrime**2)
        return 1 - 2*eta*r/(1-r+eta)  # polar scattering angle mu

    def evalStoppingPower(self, Ekin: float, material: Material) -> float:
        """ Stopping power PENELOPE found in Olbrant.

            See abstract base class method for arguments and return value.
        Note on EGSnrc stopping power.
            - A previous version implemented the stopping power from EGSnrc. In that formula I assumed the scattering center density n was equal to the number density of water. This was probably wrong. Comparing with PENELOPEs stopping power, n = NB_DENSITY*Z. CHANGE THIS!
            - I don't understand the Tc parameter in the formula from EGSnrc.
            - This previous implementation did not include density effect correction that takes into account the polarization of the medium due to the electron field.
        """
        # Ekin = tau in papers
        I = material.I
        NB_DENSITY = material.NB_DENSITY
        Z = material.Z

        Ekin_eV: float = Ekin*ERE*1e6  # Electron kinetic energy in eV (E or T in literature)
        if Ekin_eV < I:
            raise ValueError('Input energy is lower than I')

        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1, 2)

        term1 = (1+betaSquared + 2*np.sqrt(1-betaSquared))*np.log(2)
        term2 = np.power((1 - np.sqrt(1-betaSquared)), 2)/8
        Lcoll: float = 2*np.pi*np.power(Re, 2)*NB_DENSITY*Z*(np.log(Ekin_eV/I) + 1 - term1 + term2)/betaSquared

        return Lcoll
