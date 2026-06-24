# %%
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils import load_example
print(dinv.__file__)

# %% Load test images
# ----------------
#
# First, let's load some test images.

dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
psf_size = (71, 71)
pupil_size = (2048, 2048)

# Next, set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(2)
torch.cuda.manual_seed(2)

# %%
# Diffraction blur generators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We also implemented diffraction blurs obtained through Fresnel theory and definition of the psf through the pupil
# plane expanded in Zernike polynomials

from deepinv.physics.generator import DiffractionBlurGenerator

diffraction_generator = DiffractionBlurGenerator(
    psf_size = psf_size, 
    num_channels = 3, 
    pupil_size = pupil_size,
    max_zernike_amplitude = 0.3,
    device = device, 
    dtype = dtype
)

# %%
# Then, to generate new filters, it suffices to call the step() function as follows:
lambdaR = 650
lambdaG = 550
lambdaB  = 450
NA = 1.51
pixel_size = 50
fc = [NA*pixel_size/lambdaR, NA*pixel_size/lambdaG, NA*pixel_size/lambdaB]

# %%
# In this case, the `step()` function returns a dictionary containing the filters,
# their pupil function and Zernike coefficients:
filters = diffraction_generator.step(batch_size=2, fc=fc, zernike_perturbation_amplitude=0.05)
psfs = filters["filter"]
pupils = filters["pupil"]
coeffs = filters["coeff"]

# Note that we use **0.2 to increase the image dynamics
plot(
    [psfs**0.5, pupils.angle()],
    suptitle="Examples of randomly generated diffraction blurs with their phase",
)


# %%
