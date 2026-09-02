r"""
PIDAL + TV prior for PET reconstruction
======================================

Demonstrates using the PIDAL (see :footcite:t:`figueiredo_restoration_2010`) scheme with a total-variation (TV) prior
for positron emission tomography (PET) reconstruction on a simulated phantom.

This method is an alternative to ADMM for non-denoising Poisson inverse problems as it provides a splitting procedure.


"""

# %%
import deepinv as dinv
from deepinv.utils.phantoms import generate_pet_phantom
import torch

# %%
# Load PET phantom and attenuation map
#

img_size = (160, 160)
device = "cuda" if torch.cuda.is_available() else "cpu"
x, attenuation = generate_pet_phantom(img_size, device=device)

# %%
# Create PET physics and simulate sinogram data
#

voxel_size = (2.0, 2.0)
gain = 1.0
physics = dinv.physics.PET(
    img_size=img_size,
    voxel_size=voxel_size,
    device=device,
    gain=gain,
    normalize=False,
)

background = None
physics.update(attenuation=attenuation, background=background)
y = physics(x)

# %%
# Define PIDAL optimizer with TV prior
pet_prior = dinv.optim.prior.TVPrior()
data_fidelity = dinv.optim.PoissonLikelihood(
    gain=gain
)
pidal = dinv.optim.PIDAL(
    data_fidelity=data_fidelity,
    prior=pet_prior,
    max_iter=50,
    stepsize=0.5,
    lambda_reg=1.0
)

# %%
# Run PIDAL reconstruction

x_pidal = pidal.forward(
    y=y,
    physics=physics
)

# %%
# Display results

dinv.utils.plot(
    [x, x_pidal],
    titles=["Ground truth", "PIDAL reconstruction"],
    figsize=(8, 4)
)