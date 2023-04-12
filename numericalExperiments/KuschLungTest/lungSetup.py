from typing import Final
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from electronTransportCode.Material import Material
from electronTransportCode.SimulationDomain import SimulationDomain
from electronTransportCode.SimOptions import SimOptions

# TODO: program initial conditions

# Constants
A_BONE: Final[float] = 1004.62  # Molecular weight of hydroxyapatite (main component of bone)
Z_BONE: Final[float] = 13.8  # Atomic number of bone. https://newton.ex.ac.uk/teaching/resources/jjm/pam2011/Lectures/X-ray%20Interaction%202.pdf
RHO_BONE: Final[float] = 1.85  # Density [g/cm^3] Kusch & Stammer
E_DENSITY_BONE: Final[float] = 5.243*1e23  # Electron density of bone from https://www.cirsinc.com/wp-content/uploads/2019/12/062M-DS-121619.pdf
SC_DENSITY_BONE: Final[float] = E_DENSITY_BONE/Z_BONE
I_BONE: Final[float] = 106.41400 # Mean excitation energy [eV]
# obtained from from https://github.com/nrc-cnrc/EGSnrc/blob/master/HEN_HOUSE/pegs4/density_corrections/compounds/bone_cortical_icrp.density

# Bone material object
BONE_MATERIAL = Material(Z_BONE, A_BONE, I_BONE, SC_DENSITY_BONE, RHO_BONE)

# Simulation domain
class LungSimulationDomain(SimulationDomain):
    def __init__(self, bins: int = 200, file: str = 'Lung.png') -> None:
        self.width = self.height = 14.5
        self.file = file
        self.bins = bins
        self.rhoMin = 0.05  # Density of black pixels (0)
        self.rhoMax = RHO_BONE  # Density of white pixels (255)
        super().__init__(0, self.width, -self.height, 0, bins, bins, None)

        # Load image
        imageGray = Image.open(file).convert('L')
        imageGrayLowRes = imageGray.resize((bins, bins), Image.ANTIALIAS)
        self.grayScaleImage = np.asarray(imageGrayLowRes)  # save pixel values as 2D numpy array

        # Store MaterialArray
        self.materialArray = np.empty(shape=(self.grayScaleImage.size, ), dtype=Material)
        for index, pixelVal in enumerate(np.nditer(self.grayScaleImage)):
            rho = self.rhoMin + (self.rhoMax - self.rhoMin)*pixelVal/255  # type: ignore
            self.materialArray[index] = Material(Z_BONE, A_BONE, I_BONE, SC_DENSITY_BONE, rho)

    def showImage(self) -> None:
        fig, ax1 = plt.subplots(figsize=(10, 4.5))
        pos = ax1.imshow(Image.fromarray(self.grayScaleImage), cmap='gray')  # type: ignore
        ax1.set_title('Lung CT Scan')
        ax1.set_xlabel('y')
        ax1.set_ylabel('z')
        nticks = 6
        ax1.set_xticks(np.linspace(0, self.bins, nticks))
        ax1.set_xticklabels(np.linspace(0, self.width, nticks))
        ax1.set_yticks(np.linspace(0, self.bins, nticks))
        ax1.set_yticklabels(np.linspace(0, -self.width, nticks))
        fig.colorbar(pos, ax=ax1)
        plt.show()

    def getMaterial(self, index: int) -> Material:
        return self.materialArray[index]

LSD = LungSimulationDomain()
LSD.showImage()