r"""
Plug-and-Play algorithm with Mirror Descent for Poisson noise inverse problems.
====================================================================================================

This is a simple example to show how to use a mirror descent Plug-and-Play algorithm for solving an inverse problem with Poisson noise.
The prior term is a RED denoising prior with DnCNN denoiser.
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import PoissonLikelihood, L2
from deepinv.optim.prior import RED
from deepinv.unfolded import unfolded_builder
from deepinv.optim import Bregman
from torchvision import transforms
from deepinv.utils.demo import load_dataset
from deepinv.models.icnn import ICNN, tiny_ICNN
from deepinv.optim import OptimIterator
from deepinv.optim.optim_iterators import MDIteration
from deepinv.optim.optim_iterators.gradient_descent import gStepGD, fStepGD
from deepinv.loss.loss import Loss


# %%
# Define Bregman potential with an ICNN
# ----------------------------------------------------------------------------------------
#

class DeepBregman(Bregman):
    r"""
    Module for the using a deep NN as Bregman potential.
    """

    def __init__(self, forw_model, conj_model = None):
        super().__init__()
        self.forw_model = forw_model
        self.conj_model = conj_model

    def fn(self, x):
        r"""
        Computes the Bregman potential.

        :param torch.Tensor x: Variable :math:`x` at which the potential is computed.
        :return: (torch.tensor) potential :math:`\phi(x)`.
        """
        return self.forw_model(x)

    def conjugate(self, x):
        r"""
        Computes the convex conjugate potential.

        :param torch.Tensor x: Variable :math:`x` at which the conjugate is computed.
        :return: (torch.tensor) conjugate potential :math:`\phi^*(y)`.
        """
        if self.conj_model is not None:
            return self.conj_model(x)
        else:
            super().conjugate(x)
        return


# %%
# Define Mirror descent in Dual Space
# ----------------------------------------------------------------------------------------
#

class DualMDIteration(OptimIterator):
    def __init__(self, **kwargs):
        super(DualMDIteration, self).__init__(**kwargs)
        self.g_step = gStepGD(**kwargs)
        self.f_step = fStepGD(**kwargs)
        self.requires_grad_g = True

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics):
        y_prev = X["est"][0]
        bregman_potential = cur_params["bregman_potential"]
        x_prev = bregman_potential.grad_conj(y_prev)
        grad = cur_params["stepsize"] * (
            self.g_step(x_prev, cur_prior, cur_params)
            + self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
        )
        
        y = y_prev - grad
        return {"est": (y,)}
    
# %%
# Define Mirror Loss
# ----------------------------------------------------------------------------------------
#

class MirrorLoss(Loss):
    def __init__(self, metric=torch.nn.MSELoss()):
        super(MirrorLoss, self).__init__()
        self.name = "mirror"
        self.metric = metric

    def forward(self, x, x_net, y, physics, model, *args, **kwargs):
        bregman_potential = model.params_algo.bregman_potential[0]
        return self.metric(bregman_potential.grad_conj(bregman_potential.grad(x_net)), x_net)

    


# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

img_size = 256
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
n_channels = 3  # 3 for color images, 1 for gray-scale images
operation = "deblurring"
num_workers = 4 if torch.cuda.is_available() else 0

# %%
# Generate a dataset of blurred images and load it.
# ----------------------------------------------------------------------------------------
# We use the Downsampling class from the physics module to generate a dataset of blurred images.

# For simplicity, we use a small dataset for training.
# To be replaced for optimal results. For example, you can use the larger "drunet" dataset.
train_dataset_name = "CBSD500"
test_dataset_name = "set3c"
# Specify the  train and test transforms to be applied to the input images.
test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)
# Define the base train and test datasets of clean images.
train_base_dataset = load_dataset(
    train_dataset_name, ORIGINAL_DATA_DIR, transform=train_transform
)
test_base_dataset = load_dataset(
    test_dataset_name, ORIGINAL_DATA_DIR, transform=test_transform
)

# Degradation parameters
noise_level_img = 1 / 40  # Poisson Noise gain
noise_level_img = 0.03

# Generate the gaussian blur operator with Poisson noise.
physics = dinv.physics.BlurFFT(
    img_size=(n_channels, img_size, img_size),
    filter=dinv.physics.blur.gaussian_blur(),
    device=device,
    #noise_model=dinv.physics.PoissonNoise(gain=noise_level_img, clip_positive = True),
    noise_model=dinv.physics.GaussianNoise(sigma = noise_level_img),
)

my_dataset_name = "demo_unfolded_sr"
n_images_max = (
    1000 if torch.cuda.is_available() else 10
)  # maximal number of images used for training
measurement_dir = DATA_DIR / train_dataset_name / operation
generated_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_base_dataset,
    test_dataset=test_base_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename=str(my_dataset_name),
)

train_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=generated_datasets_path, train=False)

# %%
# Define the PnP algorithm.
# ----------------------------------------------------------------------------------------
# The chosen algorithm is here MD (Mirror Descent).


# Select the data fidelity term, here Poisson likelihood due to the use of Poisson noise in the forward operator.
# data_fidelity = PoissonLikelihood(gain=noise_level_img)
data_fidelity = L2()

# Set up the denoising prior. Note that we use a Gaussian noise denoiser, even if the observation noise is Poisson.
prior = dinv.optim.WaveletPrior(wv="db8", level=3, device=device)

forw_bregman = tiny_ICNN(in_channels=3, dim_hidden=64, device = device)
back_bregman = tiny_ICNN(in_channels=3, dim_hidden=64, device = device)

# Set up the optimization parameters
max_iter = 5  # number of iterations

stepsize = [1] * max_iter  # stepsize of the algorithm
lamb = [1] * max_iter

params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "lambda": lamb,
    "bregman_potential": DeepBregman(forw_model = forw_bregman, conj_model = back_bregman),
}
trainable_params = [
    "lambda",
    "stepsize",
]  # define which parameters from 'params_algo' are trainable

# Logging parameters
verbose = True

# Define the unfolded trainable model.
model = unfolded_builder(
    iteration = MDIteration(F_fn=None, has_cost=False),
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
)

# training parameters
epochs = 10 if torch.cuda.is_available() else 2
learning_rate = 5e-4
train_batch_size = 2 if torch.cuda.is_available() else 1
test_batch_size = 1

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs * 0.8))

# choose supervised training loss
losses = [dinv.loss.SupLoss(), MirrorLoss()]

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Train the network
# ----------------------------------------------------------------------------------------
# We train the network using the :meth:`deepinv.Trainer` class.

trainer = dinv.Trainer(
    model,
    physics=physics,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    optimizer=optimizer,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=True,  # disable progress bar for better vis in sphinx gallery.
    wandb_vis=False,  # training visualization can be done in Weight&Bias
)

model = trainer.train()


# %%
# Test the network
# --------------------------------------------
#
#

trainer.test(test_dataloader)