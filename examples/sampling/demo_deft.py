r"""
Implementing DEFT
==================

In this tutorial, we will go over the steps in the Doob's h-transform Efficient FineTuning (DEFT) algorithm introduced in
`Denker et al. <https://openreview.net/forum?id=AKBTFQhCjm>`_ The full algorithm is implemented in
:meth:`deepinv.sampling.DEFT`.
"""

# %%
# Installing dependencies
# -----------------------
# Let us ``import`` the relevant packages,
#
# .. note::
#           We work with an image of size 128 x 128 to reduce the computational time of this example.
#           The algorithm works best with images of size 256 x 256.
#

import numpy as np
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url
from tqdm import tqdm  # to visualize progress

image_size = 128

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# DEFT requires a small fine-tuning dataset. For this, we download the CBSD68 dataset.
# We will use the first image for testing and the rest for fine-tuning.

from deepinv.datasets import CBSD68
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset

transform = T.Compose(
    [
        T.Lambda(lambda img: F.center_crop(img, min(*img._size))),
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(0.5, 0.5),  # normalize to [-1, 1]
    ]
)

dataset = CBSD68(root="CBSB68", download=True, transform=transform)
x_true = dataset[0].unsqueeze(0).to(device)

dataset = Subset(dataset, list(range(1, len(dataset))))

dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

# %%
# In this tutorial we consider image super-resolution as the inverse problem, where the forward operator is implemented
# in :meth:`deepinv.physics.Downsampling`. We consider 2x bilinear downsampling
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

sigma = 12.75 / 255.0  # noise level

physics = dinv.physics.Downsampling(
    img_size=(3, image_size, image_size),
    filter="bilinear",
    factor=2,
    padding="replicate",
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    device=device,
)

y = physics(x_true)

imgs = [y, x_true]
plot(
    imgs,
    titles=["measurement", "groundtruth"],
)


# %%
# Diffusion model loading
# -----------------------
#
# We will take a pre-trained diffusion model that was also used for the DiffPIR algorithm, namely the one trained on
# the FFHQ 256x256 dataset. Note that this means that the diffusion model was trained with human face images,
# which is very different from the image that we consider in our example. Nevertheless, we will see later on that
# ``DEFT`` generalizes sufficiently well even in such case.


model = dinv.models.DiffUNet(large_model=False).to(device)

# %%
# Define diffusion schedule
# -------------------------
#
# We will use the standard linear diffusion noise schedule. Once :math:`\beta_t` is defined to follow a linear schedule
# that interpolates between :math:`\beta_{\rm min}` and :math:`\beta_{\rm max}`,
# we have the following additional definitions:
# :math:`\alpha_t := 1 - \beta_t`, :math:`\bar\alpha_t := \prod_{j=1}^t \alpha_j`.
# The following equations will also be useful
# later on (we always assume that :math:`\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})` hereafter.)
#
# .. math::
#
#           \mathbf{x}_t = \sqrt{\beta_t}\mathbf{x}_{t-1} + \sqrt{1 - \beta_t}\mathbf{\epsilon}
#
#           \mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\mathbf{\epsilon}
#
# where we use the reparametrization trick.

num_train_timesteps = 1000  # Number of timesteps used during training


def get_betas(
    beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=num_train_timesteps
):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)

    return betas


# Utility function to let us easily retrieve \bar\alpha_t
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


betas = get_betas()

# %%
# Training the h-transform. For reduce computational times, we only train for 60 epochs, this
# should be increased for real applications.

from deepinv.sampling import DEFT

data_fidelity = L2()


deft = DEFT(
    model,
    data_fidelity,
    physics=physics,
    device=device,
    verbose=True,
    img_size=image_size,
    max_iter=100,
)

deft.fit(dataloader, num_epochs=60, save=False)


# %%
# Given the trained h-transform we can sample from the model.
# After the initial training phase, the samplnig is quite fast.

recon = deft.forward(y, physics)

x = recon / 2 + 0.5
imgs = [y, x, x_true]
plot(imgs, titles=["measurement", "model output", "groundtruth"])
