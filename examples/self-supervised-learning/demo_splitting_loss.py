r"""
Self-supervised learning with measurement splitting
===================================================

We demonstrate self-supervised learning with measurement splitting, to
train a denoiser network on the MNIST dataset. The physics here is noisy
computed tomography, as is the case in
`Noise2Inverse <https://arxiv.org/abs/2001.11801>`__. Note this example
can also be easily applied to undersampled multicoil MRI as is the case
in `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`__.

Measurement splitting constructs a ground-truth free loss
:math:`\frac{m}{m_2}\| y_2 - A_2 \inversef{y_1}{A_1}\|^2` by splitting
the measurement and the forward operator using a randomly generated
mask.

See :class:`deepinv.loss.SplittingLoss` for full details.

"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import deepinv as dinv
from deepinv.utils.demo import get_data_home
from deepinv.models.utils import get_weights_url

torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
ORIGINAL_DATA_HOME = get_data_home()


# %%
# Define loss
# ~~~~~~~~~~~
#
# Our implementation has multiple optional parameters that control how the
# splitting is to be achieved. For example, you can:
#
# -  Use ``split_ratio`` to set the ratio of pixels used in the forward
#    pass vs the loss;
# -  Define custom masking methods using a ``mask_generator`` such as
#    :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator`
#    or :class:`deepinv.physics.generator.GaussianSplittingMaskGenerator`;
# -  Use ``eval_n_samples`` to set how many realisations of the random
#    mask is used at evaluation time;
# -  Optionally disable measurement splitting at evaluation time using
#    ``eval_split_input`` (as is the case in
#    `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`__).
# -  Average over both input and output masks at evaluation time using
#    ``eval_split_output``. See :class:`deepinv.loss.SplittingLoss` for
#    details.
#
# Note that after the model has been defined, the loss must also "adapt"
# the model.
#

loss = dinv.loss.SplittingLoss(split_ratio=0.6, eval_split_input=True, eval_n_samples=5)


# %%
# Prepare data
# ~~~~~~~~~~~~
#
# We use the ``torchvision`` MNIST dataset, and use noisy tomography
# physics (with number of angles equal to the image size) for the forward
# operator.
#
# .. note::
#
#      We use a subset of the whole training set to reduce the computational load of the example.
#      We recommend to use the whole set by setting ``train_datapoints=test_datapoints=None`` to get the best results.
#

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_HOME, train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=ORIGINAL_DATA_HOME, train=False, transform=transform, download=True
)

physics = dinv.physics.Tomography(
    angles=28,
    img_width=28,
    noise_model=dinv.physics.noise.GaussianNoise(0.1),
    device=device,
)

deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir=DATA_DIR,
    train_datapoints=100,
    test_datapoints=10,
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

train_dataloader = DataLoader(train_dataset, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=False)


# %%
# Define model
# ~~~~~~~~~~~~
#
# We use a simple U-Net architecture with 2 scales as the denoiser
# network.
#
# To reduce training time, we use a pretrained model. Here we demonstrate
# training with 100 images for 1 epoch, after having loaded a pretrained
# model trained that was with 1000 images for 20 epochs.
#
# .. note::
#
#      When using the splitting loss, the model must be "adapted" by the loss, as its forward pass takes only a subset of the pixels, not the full image.
#

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(in_channels=1, out_channels=1, scales=2).to(device), pinv=True
)
model = loss.adapt_model(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# Load pretrained model
file_name = "demo_measplit_mnist_tomography.pth"
url = get_weights_url(model_name="measplit", file_name=file_name)
ckpt = torch.hub.load_state_dict_from_url(
    url, map_location=lambda storage, loc: storage, file_name=file_name
)

model.load_state_dict(ckpt["state_dict"])
optimizer.load_state_dict(ckpt["optimizer"])


# %%
# Train and test network
# ----------------------
#

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=1,
    losses=loss,
    optimizer=optimizer,
    device=device,
    train_dataloader=train_dataloader,
    plot_images=False,
    save_path=None,
    verbose=True,
    show_progress_bar=False,
    no_learning_method="A_dagger",  # use pseudo-inverse as no-learning baseline
)

model = trainer.train()


# %%
# Test and visualise the model outputs using a small test set. We set the
# output to average over 5 iterations of random mask realisations. The
# trained model improves on the no-learning reconstruction by ~7dB.
#

trainer.plot_images = True
trainer.test(test_dataloader)


# %%
# Demonstrate the effect of not averaging over multiple realisations of
# the splitting mask at evaluation time, by setting ``eval_n_samples=1``.
# We have a worse performance:
#

model.eval_n_samples = 1
trainer.test(test_dataloader)


# %%
# Furthermore, we can disable measurement splitting at evaluation
# altogether by setting ``eval_split_input`` to False (this is done in
# `SSDU <https://pubmed.ncbi.nlm.nih.gov/32614100/>`__). This generally is
# worse than MC averaging:
#

model.eval_split_input = False
trainer.test(test_dataloader)
