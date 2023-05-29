r"""
Super-resolution with GSPnP RED.
====================================================================================================

Hurault, S., Leclaire, A., & Papadakis, N. 
Gradient Step Denoiser for convergent Plug-and-Play. 
In International Conference on Learning Representations.
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import ScoreDenoiser
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
from deepinv.training_utils import test
from torchvision import transforms
from deepinv.utils.parameters import get_GSPnP_params
from deepinv.utils.demo import load_dataset, load_degradation

# %%
# Setup paths for data loading and results.
# --------------------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure
# the reproducibility of the example.
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# --------------------------------------------------------------------------------
# In this example, we use the Set3C dataset and a motion blur kernel from
# `Levin et al. (2009) <https://ieeexplore.ieee.org/abstract/document/5206815/>`_.

dataset_name = "set3c"
img_size = 256 if torch.cuda.is_available() else 32
operation = "super-resolution"
dataset_path = ORIGINAL_DATA_DIR / dataset_name
val_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
dataset = load_dataset(dataset_name, ORIGINAL_DATA_DIR, transform=val_transform)

# Generate the degradation operator.
kernel_index = 2
kernel_torch = load_degradation(
    "kernels_12.npy", DEG_DIR / "kernels", kernel_index=kernel_index
)
kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
    0
)  # add batch and channel dimensions

# Use parallel dataloader if using a GPU to fasten training, otherwise, as all computes are on CPU, use synchronous dataloading.
num_workers = 4 if torch.cuda.is_available() else 0

factor = 2  # down-sampling factor
n_channels = 3  # 3 for color images, 1 for gray-scale images
n_images_max = 3  # Maximal number of images to restore from the input dataset
noise_level_img = 0.03  # Gaussian Noise standart deviation for the degradation
p = dinv.physics.Downsampling(
    img_size=(n_channels, img_size, img_size),
    factor=factor,
    filter=kernel_torch,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)
# Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5'" and load it.
measurement_dir = DATA_DIR / dataset_name / operation
dinv_dataset_path = dinv.datasets.generate_dataset(
    train_dataset=dataset,
    test_dataset=None,
    physics=p,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
)
dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)

# %%
# Setup the PnP algorithm
# --------------------------------------------
# We use the proximal gradient algorithm to solve the super-resolution problem with GSPnP.
# The prior g needs to be a dictionary with specified "g" and/or proximal operator "prox_g" and/or gradient "grad_g".
# For RED image restoration, a pretrained modified DRUNet denoiser replaces "grad_g".

# Parameters of the algorithm to solve the inverse problem
early_stop = True  # Stop algorithm when convergence criteria is reached
crit_conv = "cost"  # Convergence is reached when the difference of cost function between consecutive iterates is
# smaller than thres_conv
thres_conv = 1e-5
backtracking = True  # use backtracking to automatically adjust the stepsize
use_bicubic_init = False  # Use bicubic interpolation to initialize the algorithm


# load specific parameters for GSPnP
lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(
    operation, noise_level_img, kernel_index
)

params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}

# Select the data fidelity term
data_fidelity = L2()

method = "GSPnP"
denoiser_name = "gsdrunet"
# Specify the Denoising prior
ckpt_path = CKPT_DIR / "gsdrunet.ckpt"
denoiser_spec = {
    "name": denoiser_name,
    "args": {
        "in_channels": n_channels,
        "out_channels": n_channels,
        "pretrained": str(ckpt_path) if ckpt_path.exists() else "download",
        "train": False,
        "device": device,
    },
}

denoiser = ScoreDenoiser(denoiser_spec, sigma_normalize=False)
prior = {"grad_g": denoiser, "g": denoiser.denoiser.potential}


# By default, the algorithm is initialized with the adjoint of the forward operator applied to the measurements.
# For custom initialization, we need to write a function of the measurements.
if use_bicubic_init:
    custom_init = lambda y: torch.nn.functional.interpolate(
        y, scale_factor=factor, mode="bicubic"
    )
else:
    custom_init = None

# Logging parameters
verbose = True
plot_metrics = True  # compute performance and convergence metrics along the algorithm, curved saved in RESULTS_DIR

# instantiate the algorithm class to solve the IP problem.
model = optim_builder(
    algo_name="PGD",
    prior=prior,
    g_first=True,
    data_fidelity=data_fidelity,
    params_algo=params_algo,
    early_stop=early_stop,
    max_iter=max_iter,
    crit_conv=crit_conv,
    thres_conv=thres_conv,
    backtracking=backtracking,
    return_dual=True,
    verbose=verbose,
    return_metrics=plot_metrics,
    custom_init=custom_init,
)

# %%
# Evaluate the model on the problem.
# ----------------------------------------------------
# We evaluate the PnP algorithm on the test dataset, compute the PSNR metrics and plot reconstruction results.

wandb_vis = False  # plot curves and images in Weight&Bias
plot_images = True  # plot results
save_images = True  # save images in RESULTS_DIR
batch_size = 1

dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)
test(
    model=model,
    test_dataloader=dataloader,
    physics=p,
    device=device,
    plot_images=plot_images,
    save_images=save_images,
    save_folder=RESULTS_DIR / method / operation / dataset_name,
    plot_metrics=plot_metrics,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
