r"""
Weakly Convex Ridge Regularizer
===================================================

In this example we use weakly convex ridge regularizers from `this paper <https://epubs.siam.org/doi/10.1137/23M1565243>`_ by Goujon et al. for denoising and inpainting.

"""

from deepinv.optim import RidgeRegularizer
import torch
from deepinv.utils.demo import load_url_image, get_image_url
import numpy as np
from deepinv.utils.plotting import plot
from deepinv.physics import Inpainting, Denoising, GaussianNoise

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Load test image and model
# --------------------------
# We load a test image and create the RidgeRegularizer object.

url = get_image_url("CBSD_0010.png")
x = load_url_image(url, grayscale=True).to(device)
if not torch.cuda.is_available():  # use a smaller image if no gpu is available
    x = x[:, :, 100:164, 100:164]
model = RidgeRegularizer(pretrained="../../deepinv/saved_model/weights.pt").to(device)

# %%
# Denoising
# ---------------
# We create a noisy observation. The forward method of the ridge regularizer
# solves the variational problem :math:`\frac{1}{2} \|x-y\|^2+\lambda R(x,\sigma)` with :math:`\lambda=1` and noise level :math:`\sigma`.
# Since the problem is solved iteratively, we use torch.no_grad() to avoid extensive memory usage.

noise_level = 0.1
noisy = x + noise_level * torch.randn_like(x)
with torch.no_grad():
    recon = model.prox(noisy, gamma=1.0, sigma=noise_level)

plot([x, noisy, recon], titles=["ground truth", "observation", "reconstruction"])


# %%
# Inpainting
# ---------------
# We redo the procedure with an inpainting operator :math:`A`, i.e., we use the variational problem :math:`\frac{1}{2} \|Ax-y\|^2+\lambda R(x,\sigma)` where :math:`\lambda` and :math:`\sigma` have to be tuned.
# Again the problem is solved iteratively, so we use torch.no_grad() to avoid extensive memory usage.


physics = Inpainting(
    tensor_size=x.shape[1:], mask=0.5, noise_model=GaussianNoise(0.01)
).to(device)
y = physics(x)
with torch.no_grad():
    recon = model.reconstruct(physics, y, 0.05, 1.0)
plot([x, y, recon], titles=["ground truth", "observation", "reconstruction"])
