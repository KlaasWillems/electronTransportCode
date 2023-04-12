import numpy as np
import math
from electronTransportCode.ProjectUtils import A_WATER, I_WATER, SC_DENSITY, RHO_WATER, Z_WATER, FSC, CTF


class Material:
    """Wrapper object for all material parameters relevant to radiation therapy
    """
    def __init__(self, Z: float = Z_WATER, A: float = A_WATER, I: float = I_WATER, NB_DENSITY: float = SC_DENSITY, rho: float = RHO_WATER) -> None:
        """

        Args:
            Z (float, optional): Atomic number. Defaults to Z_WATER.
            A (float, optional): Relative molar mass. Defaults to A_WATER.
            I (float, optional): [eV] Mean excitation energy. Defaults to I_WATER.
            NB_DENSITY (float, optional): [cm^-3] Scattering center number density. Defaults to NB_DENSITY_WATER.
            rho (float, optional): [g/cm^3] Mass density. Defaults to RHO_WATER.
        """
        self.Z = Z
        self.A = A
        self.I = I
        self.NB_DENSITY = NB_DENSITY
        self.rho = rho

        # bc parameter used in steplength sampling
        ZS: float = Z*(Z + 1)
        ZE: float = Z*(Z + 1)*math.log(math.pow(Z, -2/3))
        ZX: float = Z*(Z + 1)*math.log(1 + 3.34*math.pow(FSC*Z, 2))
        self.bc: float = 7821.6 * rho * ZS * np.exp(ZE/ZS)/(A * np.exp(ZX/ZS))

        self.eta0CONST: float = math.pow(FSC, 2)*math.pow(Z, 2/3)/(4*math.pow(CTF, 2))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Material):
            return self.Z == other.Z and self.A == other.A and self.I == other.I and self.NB_DENSITY == other.NB_DENSITY and self.rho == other.rho and self.bc == other.bc and self.eta0CONST == other.eta0CONST
        else:
            raise NotImplementedError()


WaterMaterial = Material()
unitDensityMaterial = Material(rho=1)
