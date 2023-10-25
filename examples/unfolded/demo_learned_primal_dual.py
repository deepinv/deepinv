r"""
Learned Primal-Dual algorithm for CT scan.
====================================================================================================
Implementation of the Unfolded Primal-Dual algorithm from 

Adler, Jonas, and Ozan Öktem. 
"Learned primal-dual reconstruction." 
IEEE transactions on medical imaging 37.6 (2018): 1322-1332.

where both the data fidelity and the prior are learned modules, distinct for each iterations.

The algorithm is used for CT reconstruction trained on random phantoms. 
The phantoms are generated on the fly during training using the odl library (https://odlgroup.github.io/odl/).
"""

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.unfolded import unfolded_builder
from deepinv.training_utils import train, test
from torchvision import transforms
from deepinv.utils.phantoms import RandomPhantomDataset, SheppLoganDataset
from deepinv.optim.optim_iterators import CPIteration, fStep, gStep
from deepinv.models import PDNet_PrimalBlock, PDNet_DualBlock
from deepinv.optim import Prior, DataFidelity

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

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load base image datasets and degradation operators.
# ---------------------------------------------------
# In this example, we use the CBSD500 dataset for training and the Set3C dataset for testing.

img_size = 128 if torch.cuda.is_available() else 32
n_channels = 1  # 3 for color images, 1 for gray-scale images
operation = "CT"

# Degradation parameters
noise_level_img = 0.05

# Generate the CT operator.
physics = dinv.physics.Tomography(
    img_width=img_size,
    angles=30,
    circle=False,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)


# %%
# Define a custom iterator for the PDNet learned primal-dual algorithm.
# ---------------------------------------------------------------------
# The iterator is a subclass of the Chambolle-Pock iterator :meth:`deepinv.optim.optim_iterators.PDIteration`.
# In PDNet, the primal (gStep) and dual (fStep) updates are directly replaced by neural networks.
# We thus redefine the fStep and gStep classes as simple proximal operators of the data fidelity and prior, respectively.
# Afterwards, both the data fidelity and the prior proximal operators are defined as trainable models.


class PDNetIteration(CPIteration):
    r"""Single iteration of learned primal dual.
    We only redefine the fStep and gStep classes.
    The forward method is inherited from the CPIteration class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepPDNet(**kwargs)
        self.f_step = fStepPDNet(**kwargs)


class fStepPDNet(fStep):
    r"""
    Dual update of the PDNet algorithm.
    We write it as a proximal operator of the data fidelity term.
    This proximal mapping is to be replaced by a trainable model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, w, cur_data_fidelity, y, *args):
        r"""
        :param torch.Tensor x: Current first variable :math:`u`.
        :param torch.Tensor w: Current second variable :math:`A z`.
        :param deepinv.optim.data_fidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data fidelity term.
        :param torch.Tensor y: Input data.
        """
        return cur_data_fidelity.prox(x, w, y)


class gStepPDNet(gStep):
    r"""
    Primal update of the PDNet algorithm.
    We write it as a proximal operator of the prior term.
    This proximal mapping is to be replaced by a trainable model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, w, cur_prior, *args):
        r"""
        :param torch.Tensor x: Current first variable :math:`x`.
        :param torch.Tensor w: Current second variable :math:`A^\top u`.
        :param deepinv.optim.prior cur_prior: Instance of the Prior class defining the current prior.
        """
        return cur_prior.prox(x, w)


# %%
# Define the trainable prior and data fidelity terms.
# ---------------------------------------------------
# Prior and data-fidelity are respectively defined as subclass of :meth:`deepinv.optim.Prior` and :meth:`deepinv.optim.DataFidelity`.
# Their proximal operators are replaced by trainable models.


class PDNetPrior(Prior):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def prox(self, x, w):
        return self.model(x, w[:, 0:1, :, :])


class PDNetDataFid(DataFidelity):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def prox(self, x, w, y):
        return self.model(x, w[:, 1:2, :, :], y)


# Unrolled optimization algorithm parameters
max_iter = 10 if torch.cuda.is_available() else 3  # number of unfolded layers

# Set up the data fidelity term. Each layer has its own data fidelity module.
data_fidelity = [
    PDNetDataFid(model=PDNet_DualBlock().to(device)) for i in range(max_iter)
]

# Set up the trainable prior. Each layer has its own prior module.
prior = [PDNetPrior(model=PDNet_PrimalBlock().to(device)) for i in range(max_iter)]

# Logging parameters
verbose = True
wandb_vis = False  # plot curves and images in Weight&Bias


# %%
# Define the training parameters.
# -------------------------------
# We use the Adam optimizer and the StepLR scheduler.

# training parameters
epochs = 10
learning_rate = 1e-3
num_workers = 4 if torch.cuda.is_available() else 0
train_batch_size = 5
test_batch_size = 1
n_iter_training = int(1e5) if torch.cuda.is_available() else 1000
n_data = 1  # number of channels in the input
n_primal = 5  # extend the primal space
n_dual = 5  # extend the dual space


# %%
# Define the model.
# -------------------------------


def custom_init(y, physics):
    x0 = physics.A_dagger(y).repeat(1, n_primal, 1, 1)
    u0 = torch.zeros_like(y).repeat(1, n_dual, 1, 1)
    return {"est": (x0, x0, u0)}


def custom_output(X):
    return X["est"][0][:, 1, :, :].unsqueeze(1)


# Define the unfolded trainable model.
model = unfolded_builder(
    iteration=PDNetIteration(),
    params_algo={"K": physics.A, "K_adjoint": physics.A_adjoint, "beta": 0.0},
    data_fidelity=data_fidelity,
    prior=prior,
    max_iter=max_iter,
    custom_init=custom_init,
    get_output=custom_output,
)

# choose optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, T_max=epochs
)

# choose supervised training loss
losses = [dinv.loss.SupLoss(metric=dinv.metric.mse())]

# %%
# Training dataset of random phantoms.
# --------------------------------------------------------

# Define the base train and test datasets of clean images.
train_dataset_name = "random_phantom"
train_dataset = RandomPhantomDataset(
    size=img_size, n_data=1, length=n_iter_training // epochs
)
test_dataset = SheppLoganDataset(size=img_size, n_data=1)

train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers
)
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers
)


# %%
# Train the network
# ----------------------------------------------------------------------------------------
# We train the network using the library's train function.

train(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=epochs,
    scheduler=scheduler,
    losses=losses,
    physics=physics,
    optimizer=optimizer,
    device=device,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    wandb_vis=wandb_vis,  # training visualization can be done in Weight&Bias
    online_measurements=True,
)

# %%
# Test the network
# --------------------------------------------
#
#

method = "learned primal-dual"
save_folder = RESULTS_DIR / method / operation
plot_images = True  # plot images. Images are saved in save_folder.
plot_metrics = True  # compute performance and convergence metrics along the algorithm, curved saved in RESULTS_DIR and shown in wandb.

test(
    model=model,
    test_dataloader=test_dataloader,
    physics=physics,
    device=device,
    plot_images=plot_images,
    save_folder=save_folder,
    verbose=verbose,
    plot_metrics=plot_metrics,
    wandb_vis=wandb_vis,  # test visualization can be done in Weight&Bias
    online_measurements=True,
)
