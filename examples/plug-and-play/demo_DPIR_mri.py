"""
DPIR method for PnP MRI reconstruction
====================================================================================================

This example shows how to extend the DPIR method to solve a magnetic resonance image reconstruction problem.
The DPIR method was initially proposed in the paper:
Zhang, K., Zuo, W., Gu, S., & Zhang, L. (2017).
Learning deep CNN denoiser prior for image restoration.
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3929-3938).
"""
import numpy as np
import torch
import deepinv as dinv
from deepinv.physics.mri import MRIMixin
from deepinv.optim.data_fidelity import L2
from deepinv.optim import optim_builder, PnP


# %%
# Load kspace data
# ----------------------------------------------------------------------------------------
# In this example, we use a kspace data from the fastMRI dataset.

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Load kspace data
kspace = torch.load( "tmp/kspace.pt", weights_only=True).to(device)
kspace = kspace.unsqueeze(0)

# %%
# To transform the kspace data to the image domain, we can use the fft.
# Note that by default, the format of MRI data in deepinverse is (B, 2, ...) where B is the batch size and the second
# dimension is the real and imaginary part of the data.
# The following function transforms the kspace data to the image domain, taking into account these conventions.

def kspace2im(kspace):
    kspace = torch.moveaxis(kspace, 1, -1)
    kspace = torch.view_as_complex(kspace)
    im = MRIMixin.ifft(kspace)
    im = torch.moveaxis(torch.view_as_real(im), -1, 1)
    return im

im = kspace2im(kspace)

dinv.utils.plot(
    {
        "kspace": torch.log10(torch.abs(kspace[:, 0:1, ...] + 1j*kspace[:, 1:2, ...])+1e-9),
        "IFFT kspace": (x := im),
    },
    cmap='viridis'
)

# %%
# We can now define our measurement operator. We will consider that the groundtruth data is the ifft of the kspace data.

x = kspace2im(kspace)
x = x / torch.abs(x[:, 0:1, ...] + 1j*x[:, 1:2, ...]).max()

rng = torch.Generator(device=device).manual_seed(0)
img_size = x.shape[-2:]

# Define the measurement operator
physics_generator = dinv.physics.generator.RandomMaskGenerator(
    img_size=img_size, acceleration=4, rng=rng, device=device
)
mask = physics_generator.step()["mask"]

physics = dinv.physics.MRI(mask=mask, img_size=img_size, device=device)

y = physics(x)
backproj = physics.A_adjoint(y)

list_imgs = [x, mask, backproj]
titles = ["Original Image", "Mask", "Backprojection"]

dinv.utils.plot(list_imgs, titles=titles, cmap='gray')

# %%
# We are now ready to implement the DPIR algorithm.
# We will use the DRUNet architecture as the denoiser.

class ComplexDenoiser(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma):
        noisy_batch = torch.cat((x[:, 0:1, ...], x[:, 1:2, ...]), 0)
        denoised_batch = self.denoiser(noisy_batch, sigma)
        denoised = torch.cat((denoised_batch[0:1, ...], denoised_batch[1:2, ...]), axis=1)
        return denoised

denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1)
denoiser = denoiser.to(device)

complex_denoiser = ComplexDenoiser(denoiser)
sigma_drunet = 0.01

with torch.no_grad():
    denoised = complex_denoiser(backproj, sigma_drunet)

list_imgs = [x, mask, backproj, denoised]
titles = ["Original Image", "Mask", "Backprojection", "Denoised Backprojection"]

dinv.utils.plot(list_imgs, titles=titles, cmap='gray')

def get_DPIR_params(noise_level_img, max_iter=8):
    r"""
    Default parameters for the DPIR Plug-and-Play algorithm.

    :param float noise_level_img: Noise level of the input image.
    """
    s1 = 49.0 / 255.0
    s2 = noise_level_img
    sigma_denoiser = np.logspace(np.log10(s1), np.log10(s2), max_iter).astype(
        np.float32
    )
    stepsize = (sigma_denoiser / max(0.01, noise_level_img)) ** 2
    lamb = 1 / 0.23
    return lamb, list(sigma_denoiser), list(stepsize), max_iter

lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(sigma_drunet, max_iter=24)
sigma_denoiser = torch.from_numpy(np.array(sigma_denoiser))

params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}

# Select the data fidelity term
data_fidelity = L2()

# Specify the denoising prior
prior = PnP(denoiser=complex_denoiser)

# instantiate the algorithm class to solve the IP problem.
algorithm = optim_builder(
    iteration="HQS",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=max_iter,
    verbose=True,
    params_algo=params_algo,
    custom_init=None
)

with torch.no_grad():
    out_dpir = algorithm(y, physics)


def center_crop(x):
    return x[..., x.shape[-2]//2-320//2:x.shape[-2]//2+320//2, x.shape[-1]//2-320//2:x.shape[-1]//2+320//2]

list_imgs = [center_crop(x), center_crop(backproj), center_crop(out_dpir)]
titles = ["Original Image", "Backprojection", "DPIR"]

dinv.utils.plot(list_imgs, titles=titles, cmap='viridis')