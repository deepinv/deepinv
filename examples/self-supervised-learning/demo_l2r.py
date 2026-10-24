r"""
Self-supervised denoising with the Learning to Recorrupt (L2R) loss.
====================================================================================================

This example shows how to train a denoiser network in a fully self-supervised way,
using only noisy images through the Learning to Recorrupt (L2R) loss
:footcite:p:`monroy2026learning`, without requiring explicit knowledge of the noise distribution.

Consider measurements :math:`y = Ax + \epsilon`, where :math:`A` is the forward
operator and :math:`\epsilon` is unknown noise. L2R avoids comparing predictions
to clean targets, which are unavailable in self-supervised settings. Instead, the
method learns a small trainable re-corruption module :math:`h` that maps the noisy
measurement :math:`y` and synthetic Gaussian noise
:math:`\omega \sim \mathcal{N}(0, I)` to a perturbation. The re-corrupted image
:math:`y_1` is constructed as follows:

.. math::

    y_1 = y + \alpha h(\omega, y),

where :math:`\alpha>0` is a fixed hyperparameter. Then, the L2R loss is defined as

.. math::

    \mathcal{L}_{\mathrm{L2R}}(R,h)
    = \mathbb{E}_{y\sim p(y)}\left[  \|AR(y_1) - y\|^2 + \frac{2}{\alpha} h(\omega, y)^{\top} (A R(y_1) )  \right],

and optimize it through the adversarial objective

.. math::

    \min_{R}\;\max_{h}\;\mathcal{L}_{\mathrm{L2R}}(R,h),

where :math:`R` is the denoiser. The denoiser is encouraged to align predictions
with noisy observations and reduce noise correlation with the input noisy image,
while the re-corruption model is trained to maximize this noise correlation. The
central idea is that the re-corruptor architecture should be expressive enough
to capture the unknown noise distribution.

.. note::

    When maximizing with respect to :math:`h`, gradients through the first
    (data-fidelity) term and :math:`AR(y_1)` are stopped.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import deepinv as dinv
from deepinv.loss.l2r import Learning2RecorruptLoss
from deepinv.models.utils import get_weights_url
from deepinv.utils import get_data_home

# %%
# Setup paths for data loading and results.
# ---------------------------------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"
ORIGINAL_DATA_DIR = get_data_home()

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_device()
print(device)

# %%
# Load base image datasets
# ----------------------------------------------------------------------------------
# In this example, we use the MNIST dataset as the base image dataset.
#

operation = "denoising"
train_dataset_name = "MNIST"

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_DIR, train=False, transform=transform, download=True
)

# %%
# Generate a dataset of noisy images
# --------------------------------------------------------------------------------------------------
#
# To build the measurements, we choose a noise model in the physics simulator.
# We use Poisson noise by default, but you can switch to Gaussian noise by setting
# ``noise_name = "gaussian"``.
#
# .. note::
#
#       We use a subset of the whole training set to reduce the computational load of the example.
#       We recommend to use the whole set by setting ``n_images_max=None`` to get the best results.

predefined_noise_models = dict(
    gaussian=dinv.physics.GaussianNoise(sigma=0.1),
    poisson=dinv.physics.PoissonNoise(gain=0.3),
)

noise_name = "poisson"  # default noise model for this demo
noise_model = predefined_noise_models[noise_name]
physics = dinv.physics.Denoising(noise_model)
operation = f"{operation}_{noise_name}_l2r"

# Use parallel dataloader if using a GPU to speed up training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 0 if torch.cuda.is_available() else 0

n_images_max = (
    100 if torch.cuda.is_available() else 5
)  # number of images used for training

measurement_dir = DATA_DIR / train_dataset_name / operation
deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=measurement_dir,
    train_datapoints=n_images_max,
    test_datapoints=n_images_max,
    num_workers=num_workers,
    dataset_filename="demo_l2r",
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

# %%
# Set up the denoiser network
# ---------------------------------------------------------------
#
# We use a small DnCNN as the denoiser network.

model = dinv.models.ArtifactRemoval(
    dinv.models.DnCNN(
        in_channels=1,
        out_channels=1,
        depth=4,
        nf=8,
        bias=False,
        pretrained=None,
    ).to(device)
)


# %%
# Set up the training parameters and re-corruption network
# -------------------------------------------------------------------------------------------
# We set :class:`deepinv.loss.l2r.Learning2RecorruptLoss` as the training loss.
# The :class:`deepinv.loss.Learning2RecorruptLoss.RecorruptorNet` used here applies a
# pointwise monotonic MLP to Gaussian noise, followed by batch normalization and a
# learnable scaling kernel. Here, ``multiplicative=True`` additionally modulates the
# output by :math:`\sqrt{y}` to model signal-dependent noise. These options can all
# be changed, or a custom re-corruptor implementing ``forward(w, y)`` can be passed
# to the loss.
#
# .. note::
#
#       L2R learns an internal trainable re-corruption network and does not require
#       explicit knowledge of the measurement noise distribution during optimization.
#
# .. note::
#
#       We use a model pretrained for 382 epochs to reduce the demo time. Similar
#       results can be obtained by training from scratch for the same duration.

epochs = 1  # choose training epochs
learning_rate = 1e-3
batch_size = 64 if torch.cuda.is_available() else 1

# choose self-supervised training loss
recorruptor = Learning2RecorruptLoss.RecorruptorNet(
    depth=5,
    hidden_features=4,
    kernel_size=1,
    multiplicative=True,
    sigma=0.4,
).to(device)
loss = Learning2RecorruptLoss(
    metric=torch.nn.MSELoss(),
    alpha=0.5,
    eval_n_samples=10,
    recorruptor=recorruptor,
)
# Adapt before loading the wrapped model checkpoint.
model = loss.adapt_model(model).to(device)


# The trainer saves both optimizer states in its checkpoints.
optimizer = dinv.training.AdversarialOptimizer(
    torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4),
    torch.optim.Adam(recorruptor.parameters(), lr=1e-3, weight_decay=1e-6),
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer.G, step_size=int(epochs * 0.8) + 1
)

# Start with a pretrained model to reduce training time.
file_name = "demo_l2r_poisson.pth"
url = get_weights_url(model_name="demo", file_name=file_name)
ckpt = torch.hub.load_state_dict_from_url(
    url, map_location=lambda storage, loc: storage, file_name=file_name
)
model.load_state_dict(ckpt["state_dict"])
# The pretrained checkpoint contains only the reconstruction optimizer state.
optimizer.G.load_state_dict(ckpt["optimizer"])


# %%
# Train the network
# --------------------------------------------
# Training and early stopping use the self-supervised loss.
# PSNR is a supervised oracle metric for monitoring only.


verbose = True  # print training information

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# Initialize the trainer
trainer = dinv.training.AdversarialTrainer(
    model=model,
    D=recorruptor,
    physics=physics,
    epochs=epochs,
    scheduler=scheduler,
    losses=loss,
    losses_d=loss.RecorruptorLoss(),
    optimizer=optimizer,
    device=device,
    metrics=dinv.metric.PSNR(),  # supervised oracle for monitoring
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    early_stop=100,
    compute_eval_losses=True,
    early_stop_on_losses=True,
    plot_images=True,
    plot_interval=100,
    save_path=str(CKPT_DIR / operation),
    verbose=verbose,
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
    optimizer_step_multi_dataset=False,
)

# Train the network
model = trainer.train()


# %%
# Test the network
# --------------------------------------------
# Evaluate the trained network on the clean test images.
#
trainer.plot_images = True  # plot images during testing
trainer.plot_interval = 1
trainer.test(test_dataloader, metrics=dinv.metric.PSNR())

# %%
# :References:
#
# .. footbibliography::
