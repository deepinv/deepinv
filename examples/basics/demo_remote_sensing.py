r"""
Remote sensing with satellite images
====================================

In this example we demonstrate remote sensing inverse problems for multispectral satellite imaging.
We will focus on pan-sharpening, i.e., recovering high-resolution multispectral images from measurement pairs of
low-resolution multispectral images and high-resolution panchromatic (single-band) images with the forward
operator :class:`deepinv.physics.Pansharpen`.

These have important applications for image restoration in environmental monitoring, urban planning, disaster recovery etc.

We provide a convenient satellite image dataset for pan-sharpening :class:`deepinv.datasets.NBUDataset` provided in the paper `A Large-Scale Benchmark Data Set for Evaluating Pansharpening Performance <https://ieeexplore.ieee.org/document/9082183>`_
which includes data from several satellites such as WorldView satellites.

For remote sensing experiments, DeepInverse provides the following:

.. seealso::

    - :class:`Hyperspectral unmixing <deepinv.physics.HyperSpectralUnmixing>`
    - :class:`Super resolution <deepinv.physics.Downsampling>`
    - :class:`Satellite imagery dataset <deepinv.datasets.NBUDataset>`
    - Metrics for multispectral data: :class:`QNR <deepinv.loss.metric.QNR>`, :class:`SpectralAngleMapper <deepinv.loss.metric.SpectralAngleMapper>`, :class:`ERGAS <deepinv.loss.metric.ERGAS>`


"""

import deepinv as dinv
import torch

# %%
# Load raw pan-sharpening measurements
# ------------------------------------
# The dataset includes raw pansharpening measurements
# containing ``(MS, PAN)`` where ``MS`` are the low-res (4-band) multispectral and ``PAN`` are the high-res
# panchromatic images. Note there are no ground truth images!
#
# .. note::
#
#   The pan-sharpening measurements are provided as a :class:`deepinv.utils.TensorList`, since
#   the pan-sharpening physics :class:`deepinv.physics.Pansharpen` is a stacked physics combining
#   :class:`deepinv.physics.Downsampling` and :class:`deepinv.physics.Decolorize`.
#   See the User Guide :ref:`physics_combining` for more information.
#
# Note, for plotting purposes we only plot the first 3 bands (RGB).
#
# Note also that the linear adjoint must assume the unknown spectral response function (SRF).
#

DATA_DIR = dinv.utils.get_data_home()
dataset = dinv.datasets.NBUDataset(DATA_DIR, return_pan=True, download=True)

y = dataset[0].unsqueeze(0)  # MS (1,4,256,256), PAN (1,1,1024,1024)

physics = dinv.physics.Pansharpen((4, 1024, 1024), factor=4)

# Pansharpen with classical Brovey method
x_hat = physics.A_dagger(y)  # shape (1,4,1024,1024)

dinv.utils.plot(
    [
        y[0][:, :3],
        y[1],  # Note this will be interpolated to match high-res image size
        x_hat[:, :3],
        physics.A_adjoint(y)[:, :3],
    ],
    titles=[
        "Input MS",
        "Input PAN",
        "Pseudo-inverse using Brovey method",
        "Linear adjoint",
    ],
    dpi=1200,
)

# Evaluate performance - note we can only use QNR as we have no GT
qnr = dinv.metric.QNR()
print(qnr(x_net=x_hat, x=None, y=y, physics=physics))


# %%
# Simulate pan-sharpening measurements
# ------------------------------------
# We can also simulate pan-sharpening measurements so that we have pairs of
# measurements and ground truth. Now, the dataset loads ground truth images ``x``.
# For the pansharpening physics, we assume a flat spectral response function,
# but this can also be jointly learned. We simulate Gaussian noise on the panchromatic images.
#

dataset = dinv.datasets.NBUDataset(DATA_DIR, return_pan=False)

x = dataset[0].unsqueeze(0)  # just MS of shape 1,4,256,256

physics = dinv.physics.Pansharpen((4, 256, 256), factor=4, srf="flat")

y = physics(x)

# Pansharpen with classical Brovey method
x_hat = physics.A_dagger(y)

# %%
# Solving pan-sharpening with neural networks
# -------------------------------------------
# The pan-sharpening physics is compatible with the rest of the DeepInverse library
# so we can solve the inverse problem using any method provided in the library.
# For example, we use here the `PanNet <https://ieeexplore.ieee.org/document/8237455/>`_ model.
#
# This model can be trained using losses such as supervised learning using :class:`deepinv.loss.SupLoss`
# or self-supervised learning using Equivariant Imaging :class:`deepinv.loss.EILoss`, which was applied to
# pan-sharpening in `Wang et al., Perspective-Equivariant Imaging: an Unsupervised Framework for Multispectral Pansharpening <https://arxiv.org/abs/2403.09327>`_
#
# For evaluation, we use the standard full-reference metrics (ERGAS, SAM) and no-reference (QNR).
#
# .. note::
#
#   This is a tiny example using 5 images. We demonstrate training for 1 epoch for speed, but you can train from scratch using 50 epochs.
#

model = dinv.models.PanNet(hrms_shape=(4, 256, 256))
x_net = model(y, physics)

# Example training loss using measurement consistency on the multispectral images
# and Stein's Unbiased Risk Estimate on the panchromatic images.
loss = dinv.loss.StackedPhysicsLoss(
    [dinv.loss.MCLoss(), dinv.loss.SureGaussianLoss(0.05)]
)

# Evaluate performance when ground-truth available
sam = dinv.metric.distortion.SpectralAngleMapper()
ergas = dinv.metric.distortion.ERGAS(factor=4)
qnr = dinv.metric.QNR()
print(sam(x_hat, x), ergas(x_hat, x), qnr(x_hat, x=None, y=y, physics=physics))

# Load optimizer and pretrained model
optimizer = torch.optim.Adam(model.parameters())

from deepinv.models.utils import get_weights_url

file_name = "demo_nbu_pansharpen.pth"
url = get_weights_url(model_name="demo", file_name=file_name)
ckpt = torch.hub.load_state_dict_from_url(
    url, map_location=lambda storage, loc: storage, file_name=file_name
)
model.load_state_dict(ckpt["state_dict"])
optimizer.load_state_dict(ckpt["optimizer"])

# Train using deepinv Trainer
from torch.utils.data import DataLoader

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=optimizer,
    losses=loss,
    metrics=[sam, ergas],
    train_dataloader=DataLoader(dataset),
    epochs=1,
    online_measurements=True,
    plot_images=False,
    compare_no_learning=True,
    no_learning_method="A_dagger",
    show_progress_bar=False,
)

trainer.train()
trainer.test(DataLoader(dataset))

# Plot results
dinv.utils.plot(
    [
        x[:, :3],
        y[0][:, :3],
        y[1],
        x_hat[:, :3],
        x_net[:, :3],
    ],
    titles=["x HRMS", "y LRMS", "y PAN", "Estimate (classical)", "Estimate (PanNet)"],
)
