from abc import ABC, abstractmethod
import math
from typing import Optional, Union, Tuple, Final
import numpy as np
import scipy.stats
from scipy.interpolate import RegularGridInterpolator
from spherical_stats import _vmf, _utils
import spherical_stats
from electronTransportCode.Material import Material
from electronTransportCode.ProjectUtils import ERE, tuple3d, mathlog2, PROJECT_ROOT


# Copy of scatterParticle from MCParticleTracer class.
def scatterParticle(cost: float, phi: float, vec3d: tuple3d) -> tuple3d:
    """Scatter a particle across polar angle (theta) and azimuthatl scattering angle phi

    Args:
        cost (float): Cosine of polar scattering angle
        phi (float): Azimuthal scattering angle in radians
        vec3d (tuple3d): Direction of travel

    Returns:
        tuple3d: Direction of travel after scattering cost and phi with respect to vec3d
    """
    sint = math.sqrt(1 - cost**2)
    cosphi = math.cos(phi)
    sinphi = math.sin(phi)

    # Rotation matrices (See penelope documentation eq. 1.131)
    if math.isclose(abs(vec3d[2]), 1.0, rel_tol=1e-14):  # indeterminate case
        sign = math.copysign(1.0, vec3d[2])
        x = sign*sint*cosphi
        y = sign*sint*sinphi
        z = sign*cost
    else:
        tempVar = math.sqrt(1-vec3d[2]**2)
        tempVar2 = vec3d[2]*cosphi
        x = vec3d[0]*cost + sint*(vec3d[0]*tempVar2 - vec3d[1]*sinphi)/tempVar
        y = vec3d[1]*cost + sint*(vec3d[1]*tempVar2 + vec3d[0]*sinphi)/tempVar
        z = vec3d[2]*cost - tempVar*sint*cosphi

    # normalized for security
    norm = math.sqrt(x**2 + y**2 + z**2)
    return np.array((x/norm, y/norm, z/norm), dtype=float)


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
    def sampleScatteringAngles(self, Ekin: float, material: Material) -> Tuple[float, float, bool]:
        """Sample new direction of travel.

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            material (Material): material of scattering medium in cell.

        Returns:
            float: polar and azimuthal scattering angle, NEW_ABS_DIR (bool): If polar and scattering angels are with respect to the absolute coordinate system (True) or compared to the previous direction of travel (False)
        """

    def sampleMSVec(self, Ekin: float, stepsize: float, material: Material, oldVec: tuple3d) -> tuple3d:
        """Sample accumulated scattering angle theta and phi

        Args:
            Ekin (float): Incoming particle kinetic energy relative to electron rest energy (tau or epsilon in literature)
            stepsize (float): [cm] path-length
            material (Material): material of scattering medium in cell.
            oldVec (tuple3d): previous direction. Used as mean of vMF distribution

        Returns:
            float: returns a new post-diffusive step normalised velocity vector
        """
        raise NotImplementedError

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

    def energyLoss(self, Ekin: float, pos3d: tuple3d, stepsize: float, material: Material) -> float:
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
        Emid = Ekin + self.evalStoppingPower(Ekin, pos3d, material)*stepsize/2
        return self.evalStoppingPower(Emid, pos3d, material)*stepsize

    def getScatteringVariance(self, energy: float, stepsize: float, material: Material) -> tuple3d:
        """Return variance on x, y and z coordinate if the particle where oriented along the z-axis.

        Args:
            energy (float): Energy of particle
            stepsize (float): Size of diffusive step
            material (Material): Material of the current cell

        Returns:
            tuple[float, float, float]: Variance on x, y and z
        """
        raise NotImplementedError

    def getMeanMu(self, energy: float, material: Material) -> float:
        """Return average cosine of scattering angle. Only needed for KDR.
        """
        raise NotImplementedError

    def getMuMoment2(self, energy: float, material: Material) -> float:
        """Return second moment around zero of cos(theta) pdf

        Args:
            energy (float): Energy of particle
            material (Material): Material of the current cell

        Returns:
            float: second moment of mu = cos(theta) around zero
        """
        raise NotImplementedError

    def getSintMoment2(self, energy: float, material: Material) -> float:
        """Return second moment around zero of sint(theta) pdf

        Args:
            energy (float): Energy of particle
            material (Material): Material of the current cell

        Returns:
            float: second moment of sin(theta) around zero
        """
        raise NotImplementedError


    @abstractmethod
    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        """Return the mean and the variance of the absolute postcollisional velocity distribution (a la fusion, no rotational dependencies between velocities are considered)

        Args:
            pos3d (tuple3d): Mean and variance can be positional dependent

        Returns:
            Tuple[tuple3d, tuple3d]: Mean and variance
        """
        raise NotImplementedError

    @abstractmethod
    def getScatteringRate(self, pos3d: tuple3d, Ekin: float, material: Material) -> float:
        """Return the scattering rate (scale parameter of the path length distribution)

        Args:
            Ekin (float): Energy
            pos3d (tuple3d): current position of particle
            material (Material): material in grid cell

        Returns:
            float: scattering rate
        """
        raise NotImplementedError

    @abstractmethod
    def getDScatteringRate(self, pos3d: tuple3d, vec3d: tuple3d, Ekin: float, material: Material) -> tuple3d:
        """Return gradient (with respect to position) of scattering rate

        Args:
            Ekin (float): Energy of particle
            pos3d (tuple3d): Current position of particle
            vec3d (tuple3d): Direction of travel of particle
            material (Material): Material of cell

        Returns:
            float: derivative
        """
        raise NotImplementedError


class PointSourceParticle(ParticleModel):
    """Particle for point source benchmark. See Kush & Stammer, Garret & Hauck and Ganapol 1999.
    """
    def __init__(self, generator: Union[np.random.Generator, None, int] = None, sigma: float = 1.0) -> None:
        super().__init__(generator)
        self.sigma: float = sigma

    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        assert self.rng is not None
        return self.rng.exponential(scale=1/self.sigma)

    def sampleScatteringAngles(self, Ekin: float, material: Material) -> Tuple[float, float, bool]:
        # isotropic scattering
        assert self.rng is not None
        mu = self.rng.uniform(low=-1, high=1)
        phi = self.rng.uniform(low=0.0, high=2*math.pi)
        return mu, phi, True  # In case of isotropic scattering, doesn't matter if boolean is true of False

    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        return 1

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        # 3D isotropic scattering: Mean is zero, variance is 1/3
        return (np.array((0.0, 0.0, 0.0), dtype=float), np.array((1/3, 1/3, 1/3), dtype=float))

    def getScatteringRate(self, pos3d: tuple3d, Ekin: float, material: Material) -> float:
        return self.sigma

    def getDScatteringRate(self, pos3d: tuple3d, vec3d: tuple3d, Ekin: float, material: Material) -> tuple3d:
        return np.array((0.0, 0.0, 0.0), dtype=float)


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
                l = 1.0 + 0.5*math.sin(pos3d[0])
            elif self.Es == '0.1*(1 + 0.5*sin(x))':
                l = 0.1*(1.0 + 0.5*math.sin(pos3d[0]))
            elif self.Es == '10*(1 + 0.5*sin(x))':
                l = 10*(1.0 + 0.5*math.sin(pos3d[0]))
            elif self.Es == '(100 + 10*sin(x))':
                l = 100.0 + 10.0*math.sin(pos3d[0])
            elif self.Es == '(10 + 5*sin(x))':
                l = 10 + 5*math.sin(pos3d[0])
            elif self.Es == '(1 + 0.5*sin(y))':
                l = 1.0 + 0.5*math.sin(pos3d[1])
            elif self.Es == '0.1*(1 + 0.5*sin(y))':
                l = 0.1*(1.0 + 0.5*math.sin(pos3d[1]))
            elif self.Es == '10*(1 + 0.5*sin(y))':
                l = 10*(1.0 + 0.5*math.sin(pos3d[1]))
            else:
                raise NotImplementedError('Invalid scattering rate.')
            return self.rng.exponential(scale=1.0/l)

    def sampleScatteringAngles(self, Ekin: float, material: Material) -> Tuple[float, float, bool]:
        # isotropic scattering
        assert self.rng is not None
        mu = self.rng.uniform(low=-1, high=1)
        phi = self.rng.uniform(low=0.0, high=2*math.pi)
        return mu, phi, True

    def evalStoppingPower(self, Ekin: float, pos3d: tuple3d, material: Material) -> float:
        if isinstance(self.sp, float) or isinstance(self.sp, int):
            return self.sp
        else:
            if self.sp == '(1 + x**2)':
                return 1 + pos3d[0]**2
            elif self.sp == '(1 + 0.05*cos(x*2*3.1415/20))':
                return 1 + 0.05*math.cos(pos3d[0]*2*3.1415/20)
            elif self.sp == '(1 + 0.5*sin(x))':
                return 1.0 + 0.5*math.sin(pos3d[0])
            elif self.sp == '0.2*(1 + E**2)':
                return (1.0 + Ekin**2)*0.2
            else:
                raise NotImplementedError('Invalid stopping power')

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        # 3D isotropic scattering: Mean is zero, variance is 1/3
        return (np.array((0.0, 0.0, 0.0), dtype=float), np.array((1/3, 1/3, 1/3), dtype=float))

    def getScatteringRate(self, pos3d: tuple3d, Ekin: float, material: Material) -> float:
        if isinstance(self.Es, float) or isinstance(self.Es, int):
            return self.Es
        else:
            if self.Es == '(1 + 0.5*sin(x))':
                return 1.0 + 0.5*math.sin(pos3d[0])
            elif self.Es == '0.1*(1 + 0.5*sin(x))':
                return 0.1*(1.0 + 0.5*math.sin(pos3d[0]))
            elif self.Es == '10*(1 + 0.5*sin(x))':
                return 10*(1.0 + 0.5*math.sin(pos3d[0]))
            elif self.Es == '(100 + 10*sin(x))':
                return 100.0 + 10.0*math.sin(pos3d[0])
            elif self.Es == '(10 + 5*sin(x))':
                return 10 + 5*math.sin(pos3d[0])
            elif self.Es == '(1 + 0.5*sin(y))':
                return 1.0 + 0.5*math.sin(pos3d[1])
            elif self.Es == '0.1*(1 + 0.5*sin(y))':
                return 0.1*(1.0 + 0.5*math.sin(pos3d[1]))
            elif self.Es == '10*(1 + 0.5*sin(y))':
                return 10*(1.0 + 0.5*math.sin(pos3d[1]))
            else:
                raise NotImplementedError('Invalid scattering rate.')

    def getDScatteringRate(self, pos3d: tuple3d, vec3d: tuple3d, Ekin: float, material: Material) -> tuple3d:
        if isinstance(self.Es, float) or isinstance(self.Es, int):
            return np.array((0.0, 0.0, 0.0), dtype=float)
        else:
            if self.Es == '(1 + 0.5*sin(x))':
                return np.array((0.5*math.cos(pos3d[0]), 0.0, 0.0), dtype=float)
            elif self.Es == '0.1*(1 + 0.5*sin(x))':
                return np.array((0.05*math.cos(pos3d[0]), 0.0, 0.0), dtype=float)
            elif self.Es == '10*(1 + 0.5*sin(x))':
                return np.array((5*math.cos(pos3d[0]), 0.0, 0.0), dtype=float)
            elif self.Es == '(100 + 10*sin(x))':
                return np.array((10.0*math.cos(pos3d[0]), 0.0, 0.0), dtype=float)
            elif self.Es == '(10 + 5*sin(x))':
                return np.array((5*math.cos(pos3d[0]), 0.0, 0.0), dtype=float)
            elif self.Es == '(1 + 0.5*sin(y))':
                return np.array((0.0, 0.5*math.cos(pos3d[1]), 0.0), dtype=float)
            elif self.Es == '0.1*(1 + 0.5*sin(y))':
                return np.array((0.0, 0.05*math.cos(pos3d[1]), 0.0), dtype=float)
            elif self.Es == '10*(1 + 0.5*sin(y))':
                return np.array((0.0, 5*math.cos(pos3d[1]), 0.0), dtype=float)
            else:
                raise NotImplementedError('Invalid scattering rate.')


class DiffusionTestParticlev2(DiffusionTestParticle):
    # Particle which is biased in the y>0 direction
    def __init__(self, generator: Union[np.random.Generator, None, int] = None, Es: Union[float, str] = 1, sp: Union[float, str] = 1) -> None:
        super().__init__(generator, Es, sp)

    def sampleScatteringAngles(self, Ekin: float, material: Material) -> Tuple[float, float, bool]:
        # polar scattering angle
        cost = np.random.uniform(low=-1, high=1)
        # azimuthal scattering angle
        phiAngle = np.random.uniform(low=0, high=math.pi)
        return cost, phiAngle, True

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        return (np.array((0.0, 0.5, 0.0), dtype=float), np.array((1/3, 1/12, 1/3), dtype=float))


class SimplifiedEGSnrcElectron(ParticleModel):
    """A simplified electron model. Soft elastic collisions are taken into account using the screened Rutherford elastic cross section.
    Energy loss is deposited continuously using the Bethe-Bloch inelastic restricted collisional stopping power. Hard-inelastic collisions and bremstrahlung are not taken into account.
    """
    def __init__(self, generator: Union[np.random.Generator, None, int] = None, scatterer: str = '3d', msDist: Optional[str] = None) -> None:
        """
            scatterer (str, optional): 3d means azimuthal scattering angle is sampled phi~U(0, 2*pi). '2d' means scattering in the yz-plane. In this case, phi is chosen from {pi/2, 3*pi/2}. Defaults to '3d'. '2d-simple' means uniform scattering in y < 0 direction.
            msDist (str, optional): 'esag' for elliptically symmetric angular Gaussian distribution, 'vmf' for von Mises Fisher distribution or 'expon' for exponential distribution of 1-cos(\thetaMS)
        """
        if scatterer == '2d' or scatterer == '3d' or scatterer == '2d-simple':
            self.scatterer: Final[str] = scatterer
        else:
            raise ValueError('scatterer argument is invalid.')
        super().__init__(generator)
        self.msDist: Final[Optional[str]] = msDist

        # Load look-up table for variance on kinetic motion. Linear interpolation of LUTs with RegularGridInterpolator from scipy.
        # TODO: Fit look-up tables using least squares
        bigLUT = np.load(PROJECT_ROOT + '/ms/data/ownlutv2.npy')
        d = np.load(PROJECT_ROOT + '/ms/data/ownlutAxesv2.npz')
        self.LUTeAxis = d['arr_0']
        self.LUTdsAxis = d['arr_1']
        self.LUTrhoAxis = d['arr_2']
        # variance on x, y and z coordinate
        self.interpPosVar = RegularGridInterpolator((self.LUTeAxis, self.LUTdsAxis, self.LUTrhoAxis), bigLUT[:, :, :, 4:7], fill_value=None, bounds_error=False)  # type: ignore

        # Load MS LUTs
        if self.msDist is not None:
            d = np.load(PROJECT_ROOT + '/ms/data/msAngleLUTTestAxes.npz')
            self.MSLUTeAxis = d['arr_0']
            self.MSLUTdsAxis = d['arr_1']
            self.MSLUTrhoAxis = d['arr_2']
            if self.msDist == 'esag':
                self.MSESAGLUT = np.load(PROJECT_ROOT + '/ms/data/msAngleLUTTest.npy')[:, :, :, 1:6]
            elif self.msDist == 'vmf':
                self.MSKappaLUT = np.load(PROJECT_ROOT + '/ms/data/msAngleLUTTest.npy')[:, :, :, 0]
                self.interpKappa = RegularGridInterpolator((self.MSLUTeAxis, self.MSLUTdsAxis, self.MSLUTrhoAxis), self.MSKappaLUT, fill_value=None, bounds_error=False)  # type: ignore
            elif self.msDist == 'expon':
                self.MSExpLUT = np.load(PROJECT_ROOT + '/ms/data/msAngleLUTTest.npy')[:, :, :, 6:8]
            else:
                raise NotImplementedError

    def getScatteringRate(self, pos3d: tuple3d, Ekin: float, material: Material) -> float:
        # total macroscopic screened Rutherford cross section
        temp = Ekin*(Ekin+2)
        eta = material.etaCONST2/temp
        betaSquared: float = temp/((Ekin+1)**2)
        return material.SigmaCONST/(betaSquared*temp*eta*(eta+1))

    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """ Sample path-length from screened Rutherford elastic scattering cross section. See EGSnrc manual by Kawrakow et al for full details.

            See abstract base class method for arguments and return value.
        """
        assert self.rng is not None
        assert Ekin > 0, f'{Ekin=}'
        SigmaSR = self.getScatteringRate(pos, Ekin, material)
        return self.rng.exponential(1/SigmaSR)  # path-length

    def sampleMSVec(self, Ekin: float, stepsize: float, material: Material, oldVec: tuple3d) -> tuple3d:
        assert self.rng is not None
        assert self.msDist is not None
        assert Ekin >= 0, f'{Ekin=}'

        # esag and vmf sampling via spherical_stats package.
        # In this case, sampling occurs via the global numpy random object (np.random), not self.rng.
        if self.scatterer == '3d' or self.scatterer == '2d':
            z = np.array([0., 0., 1.], dtype=float)
            if self.msDist == 'esag':
                eIndex = np.abs(self.MSLUTeAxis - Ekin).argmin()
                dsIndex = np.abs(self.MSLUTdsAxis-stepsize).argmin()
                rhoIndex = np.abs(self.MSLUTrhoAxis - material.rho).argmin()
                params = self.MSExpLUT[eIndex, dsIndex, rhoIndex]
                return spherical_stats._esag._rvs(params=params, size=1)[0]
            elif self.msDist == 'vmf':
                # Linearly interpolate dispersion coefficient of vMF distribution
                kappa = self.interpKappa(np.array((Ekin, stepsize, material.rho), dtype=float))[0]
                # rvs routine from spherical_stats._vmf.VMF
                rot_matrix = _utils.rotation_matrix(z, oldVec)
                return _vmf._sample(kappa, rot_matrix, size=1)[0]
            elif self.msDist == 'expon':
                # Exponential fit of 1-cos(\thetaMS)
                eIndex = np.abs(self.MSLUTeAxis - Ekin).argmin()
                dsIndex = np.abs(self.MSLUTdsAxis-stepsize).argmin()
                rhoIndex = np.abs(self.MSLUTrhoAxis - material.rho).argmin()
                loc, scale = self.MSExpLUT[eIndex, dsIndex, rhoIndex]
                mu = 1.0 - scipy.stats.expon.rvs(loc=loc, scale=scale, size=1, random_state=self.rng)[0]  # type: ignore
                phi = self.rng.uniform(low=0.0, high=2*math.pi)
                return scatterParticle(mu, phi, oldVec)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def sampleScatteringAngles(self, Ekin: float, material: Material) -> Tuple[float, float, bool]:
        """ Sample polar scattering angle from screened Rutherford elastic scattering cross section. See EGSnrc manual by Kawrakow et al for full details.

            See abstract base class method for arguments and return value.
        """
        assert self.rng is not None
        assert Ekin >= 0, f'{Ekin=}'
        # mu
        if self.scatterer == '3d' or self.scatterer == '2d':
            eta: float = material.etaCONST2/(Ekin*(Ekin+2))
            r: float = self.rng.uniform()
            mu = 1 - 2*eta*r/(1-r+eta)
            new_direction = False
        else:  # '2d-simple'
            mu = self.rng.uniform(low=-1, high=1)
            new_direction = True

        # phi
        if self.scatterer == '3d':
            phi = self.rng.uniform(low=0.0, high=2*math.pi)
        elif self.scatterer == '2d':
            temp = self.rng.uniform()
            if temp < 0.5:
                phi = math.pi/2
            else:
                phi = 3*math.pi/2
        else:  # '2d-simple'
            phi = 3*math.pi/2
        return mu, phi, new_direction

    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """ Stopping power from PENELOPE for close and distant interactions. Equation 3.120 in PENELOPE 2018 Conference precedings.
            - This previous implementation did not include density effect correction that takes into account the polarization of the medium due to the electron field.
        """
        # Ekin = tau in papers
        Ekin_eV: float = Ekin*ERE*1e6  # Electron kinetic energy in eV (E or T in literature)
        assert Ekin_eV >= material.I, f'Input energy: {Ekin_eV} is lower than I'

        betaSquared: float = Ekin*(Ekin+2)/((Ekin+1)**2)

        gamma = Ekin+1
        argument = ((Ekin_eV/material.I)**2)*((gamma+1)/2)
        if argument > 1e-320:  # Roundings errors in argument can cause ValueError in math.log
            term1 = math.log(argument)
        else:
            term1 = math.log(1e-320)
        term2 = 1 - betaSquared - ((2*gamma - 1)/(gamma**2))*mathlog2 + (((gamma-1)/gamma)**2)/8
        Lcoll = material.LcollConst*(term1 + term2)/betaSquared

        return Lcoll

    def getDScatteringRate(self, pos3d: tuple3d, vec3d: tuple3d, Ekin: float, material: Material) -> tuple3d:
        """Scattering rate \Sigma depends on the energy and the material. The material is piecewise constant in a cell. The energy depends on the stepsize s.
        Thus \grad_x(\Sigma) = d\Sigma/dE * dE/ds * vec3d
        """
        # Take small step ds to compute gradient and compute gradient using finite difference approximation
        ds = 1e-8
        deltaE = self.energyLoss(Ekin, pos3d, ds, material)
        dEds = -deltaE/ds  # Energy always decreases

        # Derivative of scattering rate with respect to energy. See getScatteringRate
        dSdE = (-2*Ekin - 2)/(Ekin**4 + 4*Ekin**3 + 4*Ekin**2)

        return dSdE*dEds*vec3d

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        """Assume particle undergoes a very large amount of collisions during the diffusive step. As a result, the stationary post-collision angular distribution is the isotropic distribution. When fstat is isotropically distributed, it remains isotropic after scattering with f_postcol.
        """
        # 3D isotropic scattering: Mean is zero, variance is 1/3
        if self.scatterer == "3d":
            return (np.array((0.0, 0.0, 0.0), dtype=float), np.array((1/3, 1/3, 1/3), dtype=float))
        elif self.scatterer == '2d':
            return (np.array((0.0, 0.0, 0.0), dtype=float), np.array((0.0, 1/3, 1/3), dtype=float))
        else:  # '2d-simple'
            return (np.array((0.0, -math.pi/4, 0.0), dtype=float), np.array((0.0, 2/3 - (math.pi**2)/16, 1/3), dtype=float))

    def energyLoss(self, Ekin: float, pos3d: tuple3d, stepsize: float, material: Material) -> float:
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
        assert Ekin*ERE*1e6 >= material.I, f'{Ekin=}'
        assert stepsize > 0, f'{stepsize=}'
        Emid = Ekin + self.evalStoppingPower(Ekin, pos3d, material)*stepsize/2
        assert Emid*ERE*1e6 >= material.I, f'{Emid=}'
        return self.evalStoppingPower(Emid, pos3d, material)*stepsize

    def getMeanMu(self, energy: float, material: Material) -> float:
        eta: float = material.etaCONST2/(energy*(energy+2))
        return eta*(eta+1)*(1/eta + 1/(eta+1) + 2*math.log(eta) - 2*math.log(eta+1))

    def getMuMoment2(self, energy: float, material: Material) -> float:
        eta: float = material.etaCONST2/(energy*(energy+2))
        return 8*eta*(eta+1)-4*eta*(2*eta+1)*(eta+1)*math.log(1 + 1/eta) + 1

    def getSintMoment2(self, energy: float, material: Material) -> float:
        eta: float = material.etaCONST2/(energy*(energy+2))
        # inverse hyperbolic cotangent formula from wikipedia
        x = 2*eta + 1
        arcoth = math.log((x+1)/(x-1))/2
        return -2*eta*(eta+1)*(4-4*(2*eta+1)*arcoth)

    def getScatteringVariance(self, energy: float, stepsize: float, material: Material) -> tuple3d:
        return self.interpPosVar(np.array((energy, stepsize, material.rho), dtype=float))[0]

class KDRTestParticle(SimplifiedEGSnrcElectron):
    """Particle that behaves like a 10 MeV particle.

    Args:
        SimplifiedEGSnrcElectron (_type_): _description_
    """
    def __init__(self, generator: Union[np.random.Generator, None, int] = None, msDist: Optional[str] = None) -> None:
        super().__init__(generator, scatterer = '3d', msDist=msDist)
        self.EFixed: Final[float] = 10/ERE

    def getScatteringRate(self, pos3d: tuple3d, Ekin: float, material: Material) -> float:
        return super().getScatteringRate(pos3d, Ekin=self.EFixed, material=material)

    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        return super().samplePathlength(Ekin=self.EFixed, pos=pos, material=material)

    def sampleScatteringAngles(self, Ekin: float, material: Material) -> Tuple[float, float, bool]:
        return super().sampleScatteringAngles(Ekin=self.EFixed, material=material)

    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        return super().evalStoppingPower(Ekin=self.EFixed, pos=pos, material=material)

    def getDScatteringRate(self, pos3d: tuple3d, vec3d: tuple3d, Ekin: float, material: Material) -> tuple3d:
        return np.array((0.0, 0.0, 0.0), dtype=float)

    def getScatteringVariance(self, energy: float, stepsize: float, material: Material) -> tuple3d:
        return super().getScatteringVariance(energy=self.EFixed, stepsize=stepsize, material=material)

    def getMeanMu(self, energy: float, material: Material) -> float:
        return super().getMeanMu(energy=self.EFixed, material=material)

    def getMuMoment2(self, energy: float, material: Material) -> float:
        return super().getMuMoment2(energy=self.EFixed, material=material)

    def energyLoss(self, Ekin: float, pos3d: tuple3d, stepsize: float, material: Material) -> float:
        return super().energyLoss(Ekin=self.EFixed, pos3d=pos3d, stepsize=stepsize, material=material)

    def getOmegaMoments(self, pos3d: tuple3d) -> Tuple[tuple3d, tuple3d]:
        raise NotImplementedError('Particle only meant to be used with KDR.')

# Only used for plotting the stopping power
class SimplifiedPenelopeElectron(ParticleModel):
    """Electron model as found in PENELOPE manual
    """
    def evalStoppingPower(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        """ Stopping power from PENELOPE for close and distant interactions. Equation 3.120 in PENELOPE 2018 Conference precedings.
        """
        # Ekin = tau in papers

        Ekin_eV = Ekin*ERE*1e6  # Electron kinetic energy in eV (E or T in literature)
        assert Ekin_eV < material.I, f'Input energy: {Ekin_eV} is lower than I'

        betaSquared: float = Ekin*(Ekin+2)/((Ekin+1)**2)

        gamma = Ekin+1
        term1 = math.log(((Ekin_eV/material.I)**2)*((gamma+1)/2))
        term2 = 1 - betaSquared - ((2*gamma - 1)/(gamma**2))*math.log(2) + (((gamma-1)/gamma)**2)/8
        Lcoll = material.LcollConst*(term1 + term2)/betaSquared

        return Lcoll

    def samplePathlength(self, Ekin: float, pos: tuple3d, material: Material) -> float:
        raise NotImplementedError

    def sampleScatteringAngles(self, Ekin: float, material: Material) -> Tuple[float, float, bool]:
        raise NotImplementedError
