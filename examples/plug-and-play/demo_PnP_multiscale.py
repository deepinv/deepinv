"""
Multiscale Plug-and-Play
====================================================================================================

In this example, multiscale Plug-and-Play is used to show its benefits
over the regular Plug-and-Play for solving the inpainting inverse problem.
First, the results of regular Plug-and-Play is shown.
In the second part of this example, a coarse inverse problem is designed,
and the solution of the coarse problem is then used by fine scale algorithm obtain the solution.
It appears the quality of the obtained solution is worth the small computation overhead introduced
by the coarse scale operations.
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.physics.wrappers import to_multiscale
from deepinv.training import test
from deepinv.utils.demo import load_dataset
from deepinv.physics import Inpainting, GaussianNoise

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"

# %%
# Define the PnP setting.
# ----------------------------------------------------------------------------------------
# In this example, we use the Set3C dataset

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

dataset_name = "set3c"
img_size = 256 if torch.cuda.is_available() else 32
set3c_img_shape = (3, img_size, img_size)  # set3c contains 256x256 rgb images
val_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.CenterCrop(img_size)]
)

dataset = load_dataset(dataset_name, transform=val_transform)
dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

# set the data-fidelity and the prior
data_fidelity = L2()
noise_level = 0.1
noise_model = GaussianNoise(sigma=noise_level)
physics = Inpainting(
    tensor_size=set3c_img_shape, mask=0.5, noise_model=noise_model, device=device
)

prior = PnP(denoiser=DRUNet(pretrained="download", device=device))

# set values of the PnP parameters
max_iter_pnp = 24
params_algo = {"stepsize": 1.0, "g_param": 0.05}


# %%
# Classical case : single scale PnP.
# ----------------------------------------------------------------------------------------

# create the iterative algorithm model
model = optim_builder(
    iteration="PGD",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=max_iter_pnp,
    params_algo=params_algo,
)

# Set the model to evaluation mode since we do not require training.
model.eval()

# run the model on chosen dataset
test(
    model=model,
    test_dataloader=dataloader,
    physics=physics,
    metrics=[dinv.metric.PSNR()],
    device=device,
    online_measurements=True,
    plot_images=True,
    plot_convergence_metrics=True,
    verbose=True,
)

# %%
# Multiscale case: use a coarse setting to initialize the fine setting.
# ----------------------------------------------------------------------------------------
# The PnP algorithm iterates on a coarse scale to obtain a first estimate.
# This estimate is then upsampled and used as initialization in the fine scale.
# As shown in the result, the reconstruction quality significantly improves.

max_iter_ml_pnp = 8


# define the function which will be used to initialize the fine setting.
def custom_init(y, physics, F_fn=None):
    p_multiscale = to_multiscale(physics, y.shape[1:], factors=(2,))
    p_multiscale.set_scale(1)
    y_coarse = p_multiscale.downsample_measurement(y)
    params_algo = {"stepsize": 1.0, "g_param": 0.05}

    model = optim_builder(
        iteration="PGD",
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=False,
        max_iter=16,
        params_algo=params_algo,
    )

    x_coarse = model(y, p_multiscale)

    # upsample coarse estimation
    x_up = p_multiscale.upsample(x_coarse)
    return {"est": [x_up]}


# define the multiscale model by setting the "custom_init" field
model = optim_builder(
    iteration="PGD",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=max_iter_ml_pnp,
    params_algo=params_algo,
    custom_init=custom_init,
)

# Set the model to evaluation mode since we do not require training.
model.eval()

# run the multiscale algorithm exactly as any other algorithms
test(
    model=model,
    test_dataloader=dataloader,
    physics=physics,
    metrics=[dinv.metric.PSNR()],
    device=device,
    online_measurements=True,
    plot_images=True,
    plot_convergence_metrics=True,
    verbose=True,
)
