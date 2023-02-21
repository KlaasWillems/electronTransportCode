import numpy as np
from electronTransportCode.ProjectUtils import A_WATER, I_WATER, NB_DENSITY_WATER, RHO_WATER, Z_WATER, FSC, CTF


class Material:
    """Wrapper object for all material parameters relevant to radiation therapy
    """
    def __init__(self, Z: float = Z_WATER, A: float = A_WATER, I: float = I_WATER, NB_DENSITY: float = NB_DENSITY_WATER, rho: float = RHO_WATER) -> None:
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
        ZE: float = Z*(Z + 1)*np.log(np.power(Z, -2/3))
        ZX: float = Z*(Z + 1)*np.log(1 + 3.34*np.power(FSC*Z, 2))
        self.bc: float = 7821.6 * rho * ZS * np.exp(ZE/ZS)/(A * np.exp(ZX/ZS))

        self.eta0CONST: float = np.power(FSC, 2)*np.power(Z, 2/3)/(4*np.power(CTF, 2))



WaterMaterial = Material()
unitDensityMaterial = Material(rho=1)
