r"""
Tour of Non-Cartesian MRI functionality in DeepInverse
======================================================

This example presents the various datasets, forward physics and models
available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:

-  Physics: :class:`deepinv.physics.MRI`,
   :class:`deepinv.physics.MultiCoilMRI`,
   :class:`deepinv.physics.DynamicMRI`
-  Datasets: raw kspace with the `FastMRI <https://fastmri.med.nyu.edu>`__ dataset
   :class:`deepinv.datasets.FastMRISliceDataset` and an in-memory easy-to-use version
   :class:`deepinv.datasets.SimpleFastMRISliceDataset`, and raw dynamic k-t-space data with the
   `CMRxRecon <https://cmrxrecon.github.io>`__ dataset.
-  Models: :class:`deepinv.models.VarNet`
   (VarNet :footcite:t:`hammernik2018learning`, E2E-VarNet :footcite:t:`sriram2020end`),
   :class:`deepinv.models.MoDL` (a simple MoDL :footcite:t:`aggarwal2018modl` unrolled model)

Contents:

1. Get started with FastMRI (singlecoil + multicoil)
2. Train an accelerated MRI with neural networks
3. Load raw FastMRI data (singlecoil + multicoil)
4. Train using raw data
5. Explore 3D MRI
6. Explore dynamic MRI

"""

# %%
import deepinv as dinv
import torch
from torch.utils.data import DataLoader

device = "cpu"#dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device=device).manual_seed(0)


# %%
# 1. Get started with Calgary 3D Brain MRI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# You can get started with our simple
# `FastMRI <https://fastmri.med.nyu.edu>`__ mini slice subsets which provide
# quick, easy-to-use, in-memory datasets which can be used for simulation
# experiments.
#
# .. important::
#
#    By using this dataset, you confirm that you have agreed to and signed the `FastMRI data use agreement <https://fastmri.med.nyu.edu/>`_.
#
# .. seealso::
#
#   Datasets :class:`deepinv.datasets.FastMRISliceDataset` :class:`deepinv.datasets.SimpleFastMRISliceDataset`
#       We provide convenient datasets to easily load both raw and reconstructed FastMRI images.
#       You can download more data on the `FastMRI site <https://fastmri.med.nyu.edu/>`_.
#
# Load mini demo knee and brain datasets (original data is 320x320 but we resize to
# 128 for speed):
#
def get_mid_planes(x):
    """Get mid axial, coronal, sagittal planes."""
    d, h, w = x.shape[-3:]
    return [x[..., d // 2, :, :], x[..., :, h // 2, :], x[..., :, :, w // 2]]

from mrinufft.trajectories import initialize_3D_floret
from mrinufft.trajectories.display import display_3D_trajectory


trajectory = initialize_3D_floret(3000, 512)
display_3D_trajectory(trajectory[::10])
img_size = (256, 218, 170)
n_coils = 12


foward_model = dinv.physics.mri.NonCartesianMRI(
    trajectory,
    img_size,
    n_coils=12,
    squeeze_dims=True,
    grad_wrt_data=False,
)
brain_dataset = dinv.datasets.Calgary3DBrainMRIDataset(
    #dinv.utils.get_data_home(),
    "/volatile/",
    download_example=True,
    transform=dinv.datasets.CalgaryDataTransformer(foward_model=foward_model),
)
kspace_data, recon_image = brain_dataset[0]
#dinv.utils.plot(get_mid_planes(recon_image))
# %%
# We can also simulate multicoil MRI data. Either pass in ground-truth
# coil maps, or pass an integer to simulate simple birdcage coil maps. The
# measurements ``y`` are now of shape ``(B, C, N, H, W)``, where ``N`` is
# the coil-dimension.
#

physics = dinv.physics.mri.NonCartesianMRI(
    trajectory,
    img_size,
    n_coils=12,
    density=True,
    smaps={"name": "low_frequency", "kspace_data": kspace_data}, # use low-res kspace to estimate maps
)
x_zf = physics.A_adjoint(kspace_data.to(device))
physics.E.density = None # disable density compensation for iterations 
#dinv.utils.plot(get_mid_planes(x_zf))


prior = dinv.optim.prior.WaveletPrior(level=3, wv="db8", wvdim=3, p=1, device=device, is_complex=True)

model = dinv.optim.optim_builder(
    iteration="FISTA",
    max_iter=10,
    prior=prior,
    early_stop=True,
    custom_init=lambda y, physics: {"est": (x_zf, x_zf.clone())},
    params_algo={"stepsize": 1/physics.E.get_lipschitz_cst(), "lambda": 10.0},
    data_fidelity=dinv.optim.data_fidelity.L2(),
    verbose=True,
)
x_hat = model(kspace_data.to(device), physics=physics)
dinv.utils.plot(get_mid_planes(x_hat))

# %%
# These backbones can be used within specific MRI models, such as
# VarNet :footcite:t:`hammernik2018learning`, E2E-VarNet :footcite:t:`sriram2020end` and MoDL :footcite:t:`aggarwal2018modl`,
# for which we provide implementations:
#

# %%
# :References:
#
# .. footbibliography::
