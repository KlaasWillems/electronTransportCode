import math
from electronTransportCode.ProjectUtils import A_WATER, I_WATER, SC_DENSITY_WATER, RHO_WATER, Z_WATER, FSC, CTF, Re, ERE, u


class Material:
    """Wrapper object for all material parameters relevant to radiation therapy
    """
    def __init__(self, Z: float = Z_WATER, A: float = A_WATER, I: float = I_WATER, SC_DENSITY: float = SC_DENSITY_WATER, rho: float = RHO_WATER) -> None:
        """

        Args:
            Z (float, optional): Atomic number. Defaults to Z_WATER.
            A (float, optional): Relative molar mass. Defaults to A_WATER.
            I (float, optional): [eV] Mean excitation energy. Defaults to I_WATER.
            SC_DENSITY (float, optional): [cm^-3] Scattering center number density. Defaults to SC_DENSITY_WATER.
            rho (float, optional): [g/cm^3] Mass density. Defaults to RHO_WATER.
        """
        self.Z = Z
        self.A = A
        self.I = I
        self.SC_DENSITY = SC_DENSITY
        self.rho = rho

        # bc parameter used in steplength sampling
        ZS: float = Z*(Z + 1)
        ZE: float = Z*(Z + 1)*math.log(math.pow(Z, -2/3))
        ZX: float = Z*(Z + 1)*math.log(1 + 3.34*math.pow(FSC*Z, 2))
        self.bc: float = 7821.6 * rho * ZS * math.exp(ZE/ZS)/(A * math.exp(ZX/ZS))
        self.X: float = 0.1569 * rho * ZS/self.A
        self.SigmaCONST = math.pi*((Re*self.Z)**2)*self.rho/(u*self.A)
        self.etaCONST2: float = 1.13*(FSC**2)*math.exp(ZX/ZS)/(4*(CTF**2)*math.exp(ZE/ZS))
        # self.eta0CONST: float = math.pow(FSC, 2)*math.pow(Z, 2/3)/(4*math.pow(CTF, 2))
        self.LcollConst = 2*math.pi*(Re**2)*self.SC_DENSITY*self.Z

WaterMaterial = Material()
unitDensityMaterial = Material(rho=1)
