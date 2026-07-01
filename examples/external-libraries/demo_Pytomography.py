r"""
Single Photon Emission Computed Tomography (SPECT) reconstruction with attenuation and PSF modeling using Pytomography
===================================================

In this example we show how to use the :class:`deepinv.physics.Emission_Tomography` forward model.

We will also see an easy implementation of MLEM and MAP-EM algorithms for SPECT reconstruction.
"""

# %%
import torch
from pathlib import Path
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.io.SPECT import dicom
import matplotlib.pyplot as plt

import os
import deepinv as dinv
from deepinv.physics.EmissionTomography import SPECT
from deepinv.optim.prior import WaveletPrior

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load data
# -----------------------------------------------
# We load SPECT projection data and CT images for attenuation correction from DICOM files.
# We then estimate scatter using the double energy window method provided in Pytomography. (Optional)
#

data_path = Path("../../../data/IEC_Lu177-NEMA-SymT2")

path_CT = data_path / "CT"
files_CT = [os.path.join(path_CT, file) for file in os.listdir(path_CT)]
file_NM = data_path / "projection_data.dcm"
photopeak = dicom.get_projections(file_NM, index_peak=0)
scatter = dicom.get_energy_window_scatter_estimate(
    file_NM, index_peak=0, index_lower=1, index_upper=2
)

y = photopeak.to(device)

# add batch and channel dimensions (B, C, N_proj, H', W')
y = y.unsqueeze(0).unsqueeze(0) 

# Visualize the projection data
dinv.utils.plot(y[:, :, :, :, 60], suptitle="Sinogram slice")
proj_list = [y[:, :, 0, :, :], y[:, :, 20, :, :],  y[:, :, 60, :, :], y[:, :, 90, :, :]]
dinv.utils.plot(proj_list,subtitles=["0°", "20°", "60°", "90°"], suptitle='Projection at differents angles')

# %%
# Create forward model
# -----------------------------------------------
#
# We start by creating Pytomography's metadata and transforms. Those metadata encode the
# geometry of the imaging system. The transforms are used to model physical effects such
# as attenuation and point spread function (PSF).
#
# We then create the :class:`deepinv.physics.Emission_Tomography` forward model defined as:
# .. math::
#       y ~ P(Ax + s)
#
# where :math:`P` is a Poisson noise model, :math:`A` is the system matrix modeling the physics of the imaging system,
# The operator :math:`A` can includes attenuation and point spread function (PSF) effects, :math:`x` is the
# image to reconstruct, :math:`s` is an optional additive scatter term and :math:`y` are the measured projections.
#
# :math:`x` is of shape (B, C, D, H, W) where B is the batch size, C is the number of channels (C=1 for emission tomography),
# D is the depth, H is the height and W is the width of the 3D image.
# :math:`y` is of shape (B, C, N_proj, H', W') where B is the batch size, C is the number of channels (C=1 for emission tomography),
# N_proj is the number of projections, H' is the height and W' is the width of the projections.

object_meta, proj_meta = dicom.get_metadata(file_NM, index_peak=0)
collimator_name = "SY-ME"
energy_kev = 208  # keV
intrinsic_resolution = 0.38  # mm
psf_meta = dicom.get_psfmeta_from_scanner_params(
    collimator_name, energy_kev, intrinsic_resolution=intrinsic_resolution
)

physics = SPECT(
    object_meta = object_meta,
    proj_meta = proj_meta,
    psf_meta = psf_meta,                        # Here attenuation is define using the 'CT_file' variable
    CT_file = files_CT,                         # you can also use the 'attenuation' variable that expect 
    device = device,                             # a torch tensor. 
    noise_model=dinv.physics.PoissonNoise()    
)

# %%
# MLEM reconstruction
# -----------------------------------------------
# We implement the MLEM algorithm for SPECT reconstruction. Also called the Richardson-Lucy algorithm
# in the image deconvolution literature, the MLEM algorithm is an iterative algorithm that maximizes the
# likelihood of the measured projections under a Poisson noise model. The MLEM update is given by:
# .. math::
#       x^{k+1} = \frac{x^k}{A^T 1} A^T \left( \frac{y}{Ax^k + s} \right)
#


def mlem(x0, y, it, physics, scatter=0):
    x = x0
    At1 = physics.A_adjoint(torch.ones_like(y))
    for i in range(it):
        Ax = physics.A(x)
        ratio = y / (Ax + scatter + 1e-6)
        x = x / At1 * physics.A_adjoint(ratio)
    return x

x_ones = torch.ones(object_meta.shape).to(device)  # create an initial guess of ones
x_ones = x_ones.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
x_mlem = mlem(x_ones, y, 60, physics)
# %%
dinv.utils.plot(x_mlem[:, :, :, :, 60].swapaxes(2,3), suptitle="reconstructed object", figsize = (5,5))

# %%
# MAP-EM OSL reconstruction
# -----------------------------------------------
# We implement the MAP-EM One-Step-Late algorithm for SPECT reconstruction with a wavelet prior.
# See :footcite:t:`green1990bayesian` for more details on this algorithm.
#
# The MAP-EM update is given by:
# .. math::
#       x^{k+1} = \frac{x^k}{A^T 1 + \beta \nabla R(x^k)} A^T \left( \frac{y}{Ax^k + s} \right)
#


def mapem(x0, y, it, physics, prior, beta, scatter=0):
    x = x0
    At1 = physics.A_adjoint(torch.ones_like(y))
    for _ in range(it):
        Ax = physics.A(x)
        ratio = y / (Ax + scatter + 1e-6)
        x = x / (At1 + beta * prior.grad(x)) * physics.A_adjoint(ratio)
    return x


prior = WaveletPrior(wv="db4", device=device, wvdim=3)
x_mapem = mapem(x_ones, y, 60, physics, prior, 0.3)

# %%
dinv.utils.plot(x_mapem[:, :, :, :, 60].swapaxes(2,3), suptitle="reconstructed object", figsize = (5,5))

# %%
