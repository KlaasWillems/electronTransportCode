from electronTransportCode.ProjectUtils import A_WATER, I_WATER, NB_DENSITY_WATER, RHO_WATER, Z_WATER


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


WaterMaterial = Material()
unitDensityMaterial = Material(rho=1)
