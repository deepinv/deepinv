r"""
Check colour PSF modifs
===================================================
"""

# %%
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils import load_example

# %% Load test images
# ----------------
#
# First, let's load some test images.

dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = (256, 300)
psf_size = (71, 71)
pupil_size = (2048, 2048)

B, C = 5, 3

x_rgb = load_example(
    "CBSD_0010.png", grayscale=False, device=device, dtype=dtype, img_size=img_size
)

# Next, set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# %%
# Diffraction blur generators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We also implemented diffraction blurs obtained through Fresnel theory and definition of the psf through the pupil
# plane expanded in Zernike polynomials

from deepinv.physics.generator import DiffractionBlurGenerator

diffraction_generator = DiffractionBlurGenerator(
    psf_size = psf_size, 
    num_channels = C, 
    pupil_size = pupil_size,
    pixel_size = 50,
    NA = 1.51,
    lambda_ref = 450,
    max_zernike_amplitude = 0.3,
    zernike_index = (8,),
    device = device, 
    dtype = dtype
)

# %%
# Then, to generate new filters, it suffices to call the step() function as follows:
lambda_min = 350
lambda_max = 650
wavelengths = torch.rand(B, C) * (lambda_max - lambda_min) + lambda_min
wavelengths = wavelengths.to(device=device, dtype=dtype)

filters = diffraction_generator.step(batch_size=B, wavelengths= wavelengths)

# %%
# In this case, the `step()` function returns a dictionary containing the filters,
# their pupil function and Zernike coefficients:
print(filters.keys())

# Note that we use **0.2 to increase the image dynamics
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="Examples of randomly generated diffraction blurs",
)
plot(
    [
        f
        for f in torch.angle(filters["pupil"])
        * torch.abs(filters["pupil"])
    ],
    suptitle="Corresponding pupil phases",
)


#%%
psfs = filters["filter"].view(B*C, 1, *diffraction_generator.psf_size[-2:])
pupils = filters["pupil"].view(B*C, 1, *diffraction_generator.pupil_size[-2:])


plot(
    [
        f**0.5
        for f in psfs
    ],
    suptitle="PSFs",
    # titles=[f"lambda = {w.item():.0f}" for w in wavelengths.view(B*C)],
    fontsize=10,
    cmap="magma",
)



plot(
    [
        f
        for f in torch.angle(pupils)
        * torch.abs(pupils)
    ],
    suptitle="Corresponding pupil phases",
    # titles=[f"lambda = {w.item():.0f}" for w in w?avelengths.view(B*C)],
    fontsize=10,
    cmap="magma",
)



print("Coefficients of the decomposition on Zernike polynomials")
print(filters["coeff"])


fcs = diffraction_generator.NA * diffraction_generator.pixel_size / wavelengths
print(f"fc = {fcs}")














# %%
# We can change the cutoff frequency (below 1/4 to respect Shannon's sampling theorem)
diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size), fc=1 / 8, device=device, dtype=dtype
)
filters = diffraction_generator.step(batch_size=3)
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="A different cutoff frequency",
)

# %%
# We provide a helper property to get the list of Zernike polynomials used in the decomposition:
zernike_polynomials = diffraction_generator.zernike_polynomials
print("Zernike polynomials used:\n", "\n ".join(zernike_polynomials))

# It is also possible to directly specify the Zernike decomposition.
# For instance, if the pupil is null, the PSF is the Airy pattern
n_zernike = len(
    zernike_polynomials
)  # number of Zernike coefficients in the decomposition
filters = diffraction_generator.step(coeff=torch.zeros(3, n_zernike, device=device))
plot(
    [f for f in filters["filter"][:, None] ** 0.3],
    suptitle="Airy pattern",
)

# %%
# Finally, notice that you can activate the aberrations you want in the ANSI/Noll
# nomenclature https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices
diffraction_generator = DiffractionBlurGenerator(
    (psf_size, psf_size),
    fc=1 / 8,
    zernike_index=(
        5,
        6,
    ),  # or equivalently zernike_index = ((2, 2), (3, -3)) for (n,m) indices
    index_convention="ansi",
    device=device,
    dtype=dtype,
)
filters = diffraction_generator.step(batch_size=3)
plot(
    [f for f in filters["filter"] ** 0.5],
    suptitle="PSF obtained with astigmatism only",
)
print(
    "Zernike polynomials used:\n", "\n ".join(diffraction_generator.zernike_polynomials)
)
