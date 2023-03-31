from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import numpy as np
from electronTransportCode.Material import Material
from electronTransportCode.ProjectUtils import ERE, FSC, Re
from electronTransportCode.ProjectUtils import tuple3d


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
    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """Sample path-length.

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            pos (tuple3d): position of particle
            material (Material): material of scattering medium in cell.

        Returns:
            float: path-length [cm]
        """

    @abstractmethod
    def sampleNewVec(self, pos: tuple3d, vec: tuple3d, Ekin: float, material: Material) -> tuple3d:
        """Sample new direction of travel.

        Args:
            pos (tuple3d): position of particle
            vec (tuple3d): direction of travel of particle
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            material (Material): material of scattering medium in cell.
            NEW_ABS_DIR (bool): If polar and scattering angels are with respect to the absolute coordinate system or compared to the previous direction of travel

        Returns:
            float: polar scattering angle
        """

    @abstractmethod
    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """Evaluate electron stopping power.

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            pos (tuple3d): position of particle
            material (Material): material of scattering medium in cell.

        Returns:
            float: Stopping power evaluated at Ekin and DeltaE [1/cm] (energy relative to electron rest energy)
        """

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        """Return the mean and the variance of the postcollisional velocity distribution.

        Args:
            pos3d (tuple3d): Mean and variance can be positional dependent

        Returns:
            Tuple[tuple3d, tuple3d]: Mean and variance
        """
        raise NotImplementedError

    def getScatteringRate(self, pos3d: tuple3d) -> float:
        """Return the scattering rate (scale parameter of the path length distribution)

        Args:
            pos3d (tuple3d): current position of particle

        Returns:
            float: scattering rate
        """
        raise NotImplementedError

    def getDScatteringRate(self, pos3d: tuple3d) -> float:
        """Return derivative of scattering rate with respect to x at position pos3d

        Args:
            pos3d (tuple3d): Current position of particle

        Returns:
            float: derivative
        """
        raise NotImplementedError

    def getDirection(self, cost: float, phi: float, vec3d: tuple3d, NEW_ABS_DIR: bool = False) -> tuple3d:
        sint = np.sqrt(1 - cost**2)  # scatter left or right with equal probability
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)

        new_vec3d = np.array((0.0, 0.0, 0.0), dtype=float)

        # Rotation matrices (See penelope documentation eq. 1.131)
        if np.isclose(np.abs(vec3d[2]), 1.0, rtol=1e-14) or NEW_ABS_DIR:  # indeterminate case
            sign = np.sign(vec3d[2])
            new_vec3d[0] = sign*sint*cosphi
            new_vec3d[1] = sign*sint*sinphi
            new_vec3d[2] = sign*cost
        else:
            tempVar = np.sqrt(1-np.power(vec3d[2], 2))
            new_vec3d[0] = vec3d[0]*cost + sint*(vec3d[0]*vec3d[2]*cosphi - vec3d[1]*sinphi)/tempVar
            new_vec3d[1] = vec3d[1]*cost + sint*(vec3d[1]*vec3d[2]*cosphi + vec3d[0]*sinphi)/tempVar
            new_vec3d[2] = vec3d[2]*cost - tempVar*sint*cosphi

        # normalized for security
        new_vec3d = new_vec3d/np.linalg.norm(new_vec3d)
        return new_vec3d


class PointSourceParticle(ParticleModel):
    """Particle for point source benchmark. See Kush & Stammer, Garret & Hauck and Ganapol 1999.
    """
    def __init__(self, generator: Union[np.random.Generator, None, int] = None) -> None:
        super().__init__(generator)
        self.sigma: float = 1.0

    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        assert self.rng is not None
        return self.rng.exponential(scale=1/self.sigma)

    def sampleNewVec(self, pos: tuple3d, vec: tuple3d, Ekin: float, material: Material) -> tuple3d:
        # isotropic scattering
        assert self.rng is not None
        mu = self.rng.uniform(low=-1, high=1)
        phi = self.rng.uniform(low=0.0, high=2*np.pi)
        return self.getDirection(mu, phi, vec)

    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        return 1

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        # 3D isotropic scattering: Mean is zero, variance is 1/3
        return (np.array((0.0, 0.0, 0.0), dtype=float), np.array((1/3, 1/3, 1/3), dtype=float))

    def getScatteringRate(self, pos3d: tuple3d) -> float:
        return self.sigma

    def getDScatteringRate(self, pos3d: tuple3d) -> float:
        return 0.0


class DiffusionTestParticle(ParticleModel):
    """Particle for diffusion limit test case.
    """
    def __init__(self, generator: Union[np.random.Generator, None, int] = None, Es: Union[float, str] = 1.0, sp: Union[float, str] = 1.0) -> None:
        """-
        Args:
            generator (Union[np.random.Generator, None, int], optional): random number generator object. Defaults to None.
            Es (Union[float, str], optional): Scattering rate. Either a constant or a string for energy and positional dependent scattering rates. Defaults to 1.0.
            sp (Union[float, str], optional): Stopping power. Either a constant or a string for energy and positional dependent stopping powers. Defaults to 1.0.
        """
        super().__init__(generator)
        self.Es = Es
        self.sp = sp

    def samplePathlength(self, Ekin: float, pos3d: tuple3d, material: Material) -> float:
        assert self.rng is not None
        if isinstance(self.Es, float) or isinstance(self.Es, int):
            return self.rng.exponential(scale=1.0/self.Es)
        else:
            if self.Es == '(1 + 0.5*sin(x))':
                l = 1.0 + 0.5*np.sin(pos3d[0])
            elif self.Es == '(100 + 10*sin(x))':
                l = 100.0 + 10.0*np.sin(pos3d[0])
            elif self.Es == '(10 + 5*sin(x))':
                l = 10 + 5*np.sin(pos3d[0])
            else:
                raise NotImplementedError('Invalid scattering rate.')
            return self.rng.exponential(scale=1.0/l)

    def sampleNewVec(self, pos: tuple3d, vec: tuple3d, Ekin: float, material: Material) -> tuple3d:
        # isotropic scattering
        assert self.rng is not None
        mu = self.rng.uniform(low=-1, high=1)
        phi = self.rng.uniform(low=0.0, high=2*np.pi)
        return self.getDirection(mu, phi, vec)

    def evalStoppingPower(self, Ekin: float, pos3d: tuple3d, material: Material) -> float:
        if isinstance(self.sp, float) or isinstance(self.sp, int):
            return self.sp
        else:
            if self.sp == '(1 + x**2)':
                return 1 + pos3d[0]**2
            elif self.sp == '(1 + 0.05*cos(x*2*3.1415/20))':
                return 1 + 0.05*np.cos(pos3d[0]*2*3.1415/20)
            elif self.sp == '(1 + 0.5*sin(x))':
                return 1.0 + 0.5*np.sin(pos3d[0])
            elif self.sp == '0.2*(1 + E**2)':
                return (1.0 + Ekin**2)*0.2
            else:
                raise NotImplementedError('Invalid stopping power')

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        # 3D isotropic scattering: Mean is zero, variance is 1/3
        return (np.array((0.0, 0.0, 0.0), dtype=float), np.array((1/3, 1/3, 1/3), dtype=float))

    def getScatteringRate(self, pos3d: tuple3d) -> float:
        if isinstance(self.Es, float) or isinstance(self.Es, int):
            return self.Es
        else:
            if self.Es == '(1 + 0.5*sin(x))':
                return 1.0 + 0.5*np.sin(pos3d[0])
            elif self.Es == '(100 + 10*sin(x))':
                return 100.0 + 10.0*np.sin(pos3d[0])
            elif self.Es == '(10 + 5*sin(x))':
                return 10 + 5*np.sin(pos3d[0])
            else:
                raise NotImplementedError('Invalid scattering rate.')

    def getDScatteringRate(self, pos3d: tuple3d) -> float:
        if isinstance(self.Es, float) or isinstance(self.Es, int):
            return 0.0
        else:
            if self.Es == '(1 + 0.5*sin(x))':
                return 0.5*np.cos(pos3d[0])
            elif self.Es == '(100 + 10*sin(x))':
                return 10.0*np.cos(pos3d[0])
            elif self.Es == '(10 + 5*sin(x))':
                return 5*np.cos(pos3d[0])
            else:
                raise NotImplementedError('Invalid scattering rate.')


class DiffusionTestParticlev2(DiffusionTestParticle):
    # Particle which is biased in the x>0 direction
    def __init__(self, generator: Union[np.random.Generator, None, int] = None, Es: Union[float, str] = 1, sp: Union[float, str] = 1) -> None:
        super().__init__(generator, Es, sp)

    def sampleNewVec(self, pos: tuple3d, vec: tuple3d, Ekin: float, material: Material) -> tuple3d:
        # polar scattering angle
        cost = np.random.uniform(low=-1, high=1)
        sint = np.sqrt(1 - cost**2)

        # azimuthal scattering angle
        phiAngle = np.random.uniform(low=-np.pi/2, high=np.pi/2)
        cosphi = np.cos(phiAngle)
        sinphi = np.sin(phiAngle)
        return np.array((sint*cosphi, sint*sinphi, cost), dtype=float)

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        return (np.array((0.5, 0.0, 0.0), dtype=float), np.array((1/12, 1/3, 1/3), dtype=float))


class SimplifiedEGSnrcElectron(ParticleModel):
    """A simplified electron model. Soft elastic collisions are taken into account using the screened Rutherford elastic cross section.
    Energy loss is deposited continuously using the Bethe-Bloch inelastic restricted collisional stopping power. Hard-inelastic collisions and bremstrahlung are not taken into account.
    """

    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """ Sample path-length from screened Rutherford elastic scattering cross section. See EGSnrc manual by Kawrakow et al for full details.

            See abstract base class method for arguments and return value.
        """
        assert self.rng is not None
        assert Ekin > 0, f'{Ekin=}'
        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1,2)
        SigmaSR: float = material.bc/betaSquared  # total macroscopic screened Rutherford cross section
        return self.rng.exponential(1/SigmaSR)  # path-length

    def sampleNewVec(self, pos: tuple3d, vec: tuple3d, Ekin: float, material: Material) -> tuple3d:
        """ Sample polar scattering angle from screened Rutherford elastic scattering cross section. See EGSnrc manual by Kawrakow et al for full details.

            See abstract base class method for arguments and return value.
        """
        assert self.rng is not None
        assert Ekin > 0, f'{Ekin=}'
        Z = material.Z
        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1,2)
        beta: float = np.sqrt(betaSquared)
        alfaPrime: float = FSC*Z/beta
        eta0: float = material.eta0CONST/(Ekin*(Ekin+2))
        r: float = self.rng.uniform()
        eta: float = eta0*(1.13 + 3.76*np.power(alfaPrime, 2))

        mu = 1 - 2*eta*r/(1-r+eta)
        phi = self.rng.uniform(low=0.0, high=2*np.pi)
        return self.getDirection(mu, phi, vec, False)

    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """ Stopping power from PENELOPE for close and distant interactions.

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

        gamma = Ekin+1
        term1 = np.log(np.power(Ekin_eV/I, 2)*((gamma+1)/2))
        term2 = 1 - betaSquared - ((2*gamma - 1)/gamma)*np.log(2) + np.power((gamma-1)/gamma, 2)/8
        Lcoll: float = 2*np.pi*np.power(Re, 2)*NB_DENSITY*Z*(term1 + term2)/betaSquared

        return Lcoll


class SimplifiedPenelopeElectron(ParticleModel):
    """Electron model as found in PENELOPE manual
    """
    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """ Stopping power from PENELOPE for close and distant interactions.
        """
        # Ekin = tau in papers
        I = material.I
        NB_DENSITY = material.NB_DENSITY
        Z = material.Z

        Ekin_eV: float = Ekin*ERE*1e6  # Electron kinetic energy in eV (E or T in literature)
        if Ekin_eV < I:
            raise ValueError('Input energy is lower than I')

        betaSquared: float = Ekin*(Ekin+2)/np.power(Ekin+1, 2)

        gamma = Ekin+1
        term1 = np.log(np.power(Ekin_eV/I, 2)*((gamma+1)/2))
        term2 = 1 - betaSquared - ((2*gamma - 1)/gamma)*np.log(2) + np.power((gamma-1)/gamma, 2)/8
        Lcoll: float = 2*np.pi*np.power(Re, 2)*NB_DENSITY*Z*(term1 + term2)/betaSquared

        return Lcoll

    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        raise NotImplementedError

    def sampleNewVec(self, pos: tuple3d, vec: tuple3d, Ekin: float, material: Material) -> tuple3d:
        raise NotImplementedError
