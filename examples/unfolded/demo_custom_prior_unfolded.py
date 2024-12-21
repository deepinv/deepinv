r"""
Learned iterative custom prior
==============================

This example shows how to implement a learned unrolled proximal gradient descent algorithm with a custom prior function.
The algorithm is trained on a dataset of compressed sensing measurements of MNIST images.

"""

from pathlib import Path
import torch
from torchvision import datasets
from torchvision import transforms
import deepinv as dinv
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import Prior
from deepinv.unfolded import unfolded_builder
from deepinv.utils.demo import get_data_home

# %%
# Setup paths for data loading and results.
# -----------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"
ORIGINAL_DATA_DIR = get_data_home()

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# ---------------------------------------------------
# In this example, we use MNIST as the base dataset.

img_size = 28
n_channels = 1
operation = "compressed-sensing"
train_dataset_name = "MNIST_train"

# Generate training and evaluation datasets in HDF5 folders and load them.
train_test_transform = transforms.Compose([transforms.ToTensor()])
train_base_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=True, transform=train_test_transform, download=True
)
test_base_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=False, transform=train_test_transform, download=True
)


# %%
# Generate a dataset of compressed measurements and load it.
# ----------------------------------------------------------
# We use the compressed sensing class from the physics module to generate a dataset of highly-compressed measurements
# (10% of the total number of pixels).
#
# The forward operator is defined as :math:`y = Ax`
# where :math:`A` is a (normalized) random Gaussian matrix.


# Use parallel dataloader if using a GPU to fasten training, otherwise, as all computes are on CPU, use synchronous
# data loading.
num_workers = 4 if torch.cuda.is_available() else 0

# Generate the compressed sensing measurement operator with 10x under-sampling factor.
physics = dinv.physics.CompressedSensing(
    m=78, img_shape=(n_channels, img_size, img_size), fast=True, device=device
)
my_dataset_name = "demo_LICP"
n_images_max = (
    1000 if torch.cuda.is_available() else 200
)  # maximal number of images used for training
measurement_dir = DATA_DIR / train_dataset_name / operation
generated_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    test_datapoints=8,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=False)

# %%
# Define the unfolded Proximal Gradient algorithm.
# ------------------------------------------------
# In this example, we propose to minimise a function of the form
#
# .. math::
#
#          \min_x \frac{1}{2} \|y - Ax\|_2^2 + \lambda\operatorname{TV}_{\text{smooth}}(x)
#
# where :math:`\operatorname{TV}_{\text{smooth}}` is a smooth approximation of TV.
# The proximal gradient iteration (see also :class:`deepinv.optim.optim_iterators.PGDIteration`) is defined as
#
#   .. math::
#           x_{k+1} = \text{prox}_{\gamma \lambda \operatorname{TV}_{\text{smooth}}}(x_k - \gamma A^T (Ax_k - y))
#
# where :math:`\gamma` is the stepsize and :math:`\text{prox}_{g}` is the proximity operator of :math:`g(x) =\operatorname{TV}_{\text{smooth}}(x)`.
#
# We first define the prior in a functional form.
# If the prior is initialized with a list of length max_iter,
# then a distinct weight is trained for each PGD iteration.
# For fixed trained model prior across iterations, initialize with a single model.


# Define the image gradient operator
def nabla(I):
    b, c, h, w = I.shape
    G = torch.zeros((b, c, h, w, 2), device=I.device).type(I.dtype)
    G[:, :, :-1, :, 0] = G[:, :, :-1, :, 0] - I[:, :, :-1]
    G[:, :, :-1, :, 0] = G[:, :, :-1, :, 0] + I[:, :, 1:]
    G[:, :, :, :-1, 1] = G[:, :, :, :-1, 1] - I[..., :-1]
    G[:, :, :, :-1, 1] = G[:, :, :, :-1, 1] + I[..., 1:]
    return G


# Define the smooth TV prior as the mse of the image finite difference.
def g(x, *args, **kwargs):
    dx = nabla(x)
    tv_smooth = torch.nn.functional.mse_loss(
        dx, torch.zeros_like(dx), reduction="sum"
    ).sqrt()
    return tv_smooth


# Define the prior. A prior instance from :class:`deepinv.priors` can be simply defined with an explicit potential :math:`g` function as such:
prior = Prior(g=g)

# %%
# We use :func:`deepinv.unfolded.unfolded_builder` to define the unfolded algorithm
# and set both the stepsizes of the PGD algorithm :math:`\gamma` (``stepsize``) and the soft
# thresholding parameters :math:`\lambda` as learnable parameters.
# These parameters are initialized with a table of length max_iter,
# yielding a distinct ``stepsize`` and ``g_param`` value for each iteration of the algorithm.
# For single ``stepsize`` and ``g_param`` shared across iterations, initialize with a single float value.

# Unrolled optimization algorithm parameters
max_iter = 5  # Number of unrolled iterations
lamb = [
    1.0
] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
stepsize = [
    1.0
] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.
params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "lambda": lamb,
}
trainable_params = [
    "stepsize",
    "lambda",
]  # define which parameters from 'params_algo' are trainable

# Select the data fidelity term
data_fidelity = L2()

# Logging parameters
verbose = True
wandb_vis = False  # plot curves and images in Weight&Bias

# Define the unfolded trainable model.
model = unfolded_builder(
    iteration="PGD",
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
    g_first=True,
)

# %%
# Define the training parameters.
# -------------------------------
# We now define training-related parameters,
# number of epochs, optimizer (Adam) and its hyperparameters, and the train and test batch sizes.


# Training parameters
epochs = 20 if torch.cuda.is_available() else 10
learning_rate = 5e-3  # reduce this parameter when using more epochs

# Choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

# Choose supervised training loss
losses = [dinv.loss.SupLoss(metric=torch.nn.L1Loss())]

# Batch sizes and data loaders
train_batch_size = 64 if torch.cuda.is_available() else 8
test_batch_size = 64 if torch.cuda.is_available() else 8

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Train the network.
# ------------------
# We train the network using the library's train function.
#

trainer = dinv.Trainer(
    model,
    physics=physics,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=epochs,
    device=device,
    losses=losses,
    optimizer=optimizer,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
    wandb_vis=wandb_vis,  # training visualization can be done in Weight&Bias
)


model = trainer.train()

# %%
# Test the network.
# -----------------
#
# We now test the learned unrolled network on the test dataset. In the plotted results, the `Linear` column shows the
# measurements back-projected in the image domain, the `Recons` column shows the output of our LISTA network,
# and `GT` shows the ground truth.
#

trainer.test(test_dataloader)

test_sample, _ = next(iter(test_dataloader))
model.eval()
test_sample = test_sample.to(device)

# Get the measurements and the ground truth
y = physics(test_sample)
with torch.no_grad():
    rec = model(y, physics=physics)

backprojected = physics.A_adjoint(y)

dinv.utils.plot(
    [backprojected, rec, test_sample],
    titles=["Linear", "Reconstruction", "Ground truth"],
    suptitle="Reconstruction results",
)


# %%
# Plotting the weights of the network.
# ------------------------------------
#
# We now plot the weights of the network that were learned and check that they are different from their initialization
# values. Note that ``g_param`` corresponds to :math:`\lambda` in the proximal gradient algorithm.
#

dinv.utils.plotting.plot_parameters(
    model, init_params=params_algo, save_dir=RESULTS_DIR / "unfolded_pgd" / operation
)
