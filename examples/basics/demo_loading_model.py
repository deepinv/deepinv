r"""
Training a reconstruction network.
====================================================================================================

This example shows how to train a simple reconstruction network for an image
inpainting inverse problem.

"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv
from deepinv.utils.demo import load_dataset
from deepinv.optim.data_fidelity import IndicatorL2
from deepinv.optim.prior import PnP
from deepinv.unfolded import Unfolded
from deepinv.models.denoiser import Denoiser
from deepinv.training_utils import train, test

# %%
# Setup paths for data loading and results.
# --------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# --------------------------------------------------------------------------------------------
# In this example, we use the CBSD68 dataset for training and the set3c dataset for testing.
# We work with images of size 32x32 if no GPU is available, else 128x128.


operation = "inpainting"
train_dataset_name = "CBSD68"
test_dataset_name = "set3c"
img_size = 128 if torch.cuda.is_available() else 32

test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)

train_dataset = load_dataset(train_dataset_name, ORIGINAL_DATA_DIR, train_transform)
test_dataset = load_dataset(test_dataset_name, ORIGINAL_DATA_DIR, test_transform)

# %%
# Define forward operator and generate dataset
# --------------------------------------------------------------------------------------------
# We define an inpainting operator that randomly masks pixels with probability 0.5.
#
# A dataset of pairs of measurements and ground truth images is then generated using the
# :meth:`dinv.datasets.generate_dataset` function.
#
# Once the dataset is generated, we can load it using the :class:`dinv.datasets.HDF5Dataset` class.

n_channels = 3  # 3 for color images, 1 for gray-scale images
probability_mask = 0.5  # probability to mask pixel

# Generate inpainting operator
physics = dinv.physics.Inpainting(
    (n_channels, img_size, img_size), mask=probability_mask, device=device
)


# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0
n_images_max = (
    1000 if torch.cuda.is_available() else 50
)  # maximal number of images used for training
my_dataset_name = "demo_training_inpainting"
measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)


train_batch_size = 32 if torch.cuda.is_available() else 1
test_batch_size = 32 if torch.cuda.is_available() else 1

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Set up the reconstruction network
# --------------------------------------------------------
# We use a simple inversion architecture of the form
#
#      .. math::
#
#               f_{\theta}(y) = \phi_{\theta}(A^{\top}(y))
#
# where the linear reconstruction :math:`A^{\top}y` is post-processed by a U-Net network :math:`\phi_{\theta}` is a
# neural network with trainable parameters :math:`\theta`.


# Select the data fidelity term
data_fidelity = IndicatorL2(radius=0.0)

# Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
level = 3
model_spec = {
    "name": "waveletprior",
    "args": {"wv": "db8", "level": level, "device": device},
}
# If the prior is initialized with a list of length max_iter,
# then a distinct weight is trained for each PGD iteration.
# For fixed trained model prior across iterations, initialize with a single model.
max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
prior = [PnP(denoiser=Denoiser(model_spec)) for i in range(max_iter)]

# Unrolled optimization algorithm parameters
lamb = [
    1.0
] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
stepsize = [
    1.0
] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.

sigma_denoiser_init = 0.01
sigma_denoiser = [sigma_denoiser_init * torch.ones(level, 3)] * max_iter
# sigma_denoiser = [torch.Tensor([sigma_denoiser_init])]*max_iter

stepsize = (
    0.9
    / physics.compute_norm(
        torch.ones((1, n_channels, img_size, img_size)), tol=1e-4
    ).item()
)
# stepsize = 0.9 / torch.linalg.norm(K, ord=2).item() ** 2
reg_param = 1.0
sigma = 1.0

params_algo = {
    "stepsize": stepsize,
    "g_param": reg_param,
    "lambda": lamb,
    "sigma": sigma,
    "K": physics.A,
    "K_adjoint": physics.A_adjoint,
}

trainable_params = [
    "g_param",
    "stepsize",
]  # define which parameters from 'params_algo' are trainable


def custom_init_CP(x_init, y_init):
    return {"est": (x_init, x_init, y_init)}


# Define the unfolded trainable model.
model = Unfolded(
    "CP",
    trainable_params=trainable_params,
    params_algo=params_algo,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
    g_first=False,
    custom_init=custom_init_CP,
)

# %%
# Train the model
# ----------------------------------------------------------------------------------------
# We train the model using the :meth:`dinv.training_utils.train` function.
#
# We perform supervised learning and use the mean squared error as loss function. This can be easily done using the
# :class:`dinv.loss.SupLoss` class.
#
# .. note::
#
#       In this example, we only train for a few epochs to keep the training time short.
#       For a good reconstruction quality, we recommend to train for at least 100 epochs.
#

epochs = 50 if torch.cuda.is_available() else 5  # choose training epochs
learning_rate = 5e-4

verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

# choose training losses
losses = dinv.loss.SupLoss(metric=dinv.metric.mse())

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

train(
    model=model,
    train_dataloader=train_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=physics,
    optimizer=optimizer,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,
    log_interval=1,
    eval_interval=1,
    ckp_interval=1,
)

# %%
# Test the network
# --------------------------------------------
# We can now test the trained network using the :meth:`dinv.training_utils.test` function.
#
# The testing function will compute test_psnr metrics and plot and save the results.

plot_images = True
save_images = True
method = "artifact_removal"

test_psnr, test_std_psnr, init_psnr, init_std_psnr = test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=plot_images,
    save_images=save_images,
    save_folder=RESULTS_DIR / method / operation / test_dataset_name,
    verbose=verbose,
    wandb_vis=wandb_vis,
)

# %% Saving the model and loading it
# ----------------------------------
# We can save the trained model following the standard PyTorch procedure.

# Save the model
torch.save(model.state_dict(), CKPT_DIR / operation / "model.pth")

# Load the model
# Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
level = 3
model_spec = {
    "name": "waveletprior",
    "args": {"wv": "db8", "level": level, "device": device},
}
# If the prior is initialized with a list of length max_iter,
# then a distinct weight is trained for each PGD iteration.
# For fixed trained model prior across iterations, initialize with a single model.
max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
prior_new = [PnP(denoiser=Denoiser(model_spec)) for i in range(max_iter)]

# Unrolled optimization algorithm parameters
lamb = [
    1.0
] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
stepsize = [
    1.0
] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.

sigma_denoiser_init = 0.01
sigma_denoiser = [sigma_denoiser_init * torch.ones(level, 3)] * max_iter
# sigma_denoiser = [torch.Tensor([sigma_denoiser_init])]*max_iter
params_algo_new = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
}

trainable_params = [
    "g_param",
    "stepsize",
]  # define which parameters from 'params_algo' are trainable

model_new = Unfolded(
    "FidCP",
    params_algo=params_algo_new,
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior_new,
)
model_new.load_state_dict(torch.load(CKPT_DIR / operation / "model.pth"))
model_new.eval()

# Test the model and check that the results are the same as before saving
test_psnr, test_std_psnr, init_psnr, init_std_psnr = test(
    model=model_new,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=plot_images,
    save_images=save_images,
    save_folder=RESULTS_DIR / method / operation / test_dataset_name,
    verbose=verbose,
    wandb_vis=wandb_vis,
)
