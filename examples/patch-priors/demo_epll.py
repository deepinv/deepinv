r"""
Expected Patch Log Likelihood (EPLL) for Denoising and Inpainting
====================================================================================================

In this example we use the expected patch log likelihood (EPLL) prior EPLL proposed in `"From learning models of natural image patches to whole image restoration" <https://ieeexplore.ieee.org/document/6126278>`_. 
for denoising and inpainting of natural images.
To this end, we consider the inverse problem :math:`y = Ax+\epsilon`, where :math`A` is either the identity (for denoising)
or a masking operator (for inpainting) and :math:`\epsilon\sim\mathcal{N}(0,\sigma^2 I)` is white Gaussian noise with standard deviation :math`\sigma`.
"""

from deepinv.models import EPLL
from deepinv.physics import GaussianNoise, Denoising, Inpainting
from deepinv.utils import cal_psnr, plot
import torch
import numpy as np
from deepinv.utils.demo import load_url_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Load test image and model
# ----------------------------------------
# As default EPLL loads pretrained weights for the Gaussian mixture model which where estimted based
# on 50 mio patches extracted from the BSDS500 dataset. An example how to estimate the parameters of GMM
# is included in the demo for limited-angle CT with patch priors.

url = "https://drive.google.com/uc?export=download&id=1qfRqryZdtVv86B0405Tv51wHkTmhuJhq"
test_img = load_url_image(url, grayscale=True).to(device)
patch_size = 6
model_EPLL = EPLL(patch_size=patch_size, device=device)

# %%
# Denoising
# ----------
# Define noise model, operator and generate observation

sigma = 25.0 / 255.0
noise_model = GaussianNoise(sigma)
physics = Denoising(device=device, noise_model=noise_model)
observation = physics(test_img)

r"""
Reconstruction, PSNR computation and plots.
We use the default choice of the betas in the half quadratic splitting given by
:math:`\beta \in \sigma^{-2} \{1,4,8,16,32\}`.
Generally, the betas are hyperparameters, which have to be chosen for each inverse problem seperately.
"""
x_out = model_EPLL(observation, sigma, batch_size=5000)

psnr_obs = cal_psnr(observation, test_img)
psnr_recon = cal_psnr(x_out, test_img)

print("PSNRs for Denoising:")
print("Observation: {0:.2f}".format(psnr_obs))
print("EPLL: {0:.2f}".format(psnr_recon))

plot(
    [test_img, observation.clip(0, 1), x_out.clip(0, 1)],
    ["Ground truth", "Observation", "EPLL"],
)

# %%
# Inpainting
# ----------
# Define noise model, operator and generate observation

sigma = 0.01
noise_model = GaussianNoise(sigma)
physics = Inpainting(test_img[0].shape, mask=0.7, device=device)
observation = physics(test_img)

# Reconstruction, PSNR computation and plots.
# Here, we need a different choice of beta. To this end, we extended the default choice
# by two values and optimized a constant factor via grid search.

betas = [1.0, 5.0, 10.0, 40.0, 80.0, 160.0, 320.0]
x_out = model_EPLL.reconstruction(
    observation, observation.clone(), sigma, physics, betas=betas, batch_size=5000
)

psnr_obs = cal_psnr(observation, test_img)
psnr_recon = cal_psnr(x_out, test_img)

print("PSNRs for Inpainting:")
print("Observation: {0:.2f}".format(psnr_obs))
print("EPLL: {0:.2f}".format(psnr_recon))

plot(
    [test_img, observation.clip(0, 1), x_out.clip(0, 1)],
    ["Ground truth", "Observation", "EPLL"],
)
