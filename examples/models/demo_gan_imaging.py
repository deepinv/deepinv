r"""
Training GANs with adversarial losses for imaging
=================================================

This example shows you how to train various generative adversarial networks using adversarial
learning for solving imaging inverse problems, in this case, deblurring.

We demonstrate running training and inference with the following setups:
DeblurGAN :footcite:p:`kupyn2018deblurgan`, Compressed Sensing using Generative Models (CSGM) :footcite:p:`bora2017compressed`,
AmbiantGAN :footcite:p:`bora2018ambientgan`, Unsupervised Adversarial Image Reconstruction (UAIR) :footcite:p:`pajot2019unsupervised`,
and :footcite:t:`cole2021fast`.

Adversarial networks are characterized by the addition of an adversarial loss :math:`\mathcal{L}_\text{adv}` to the standard reconstruction loss:

.. math:: \mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]

where :math:`D(\cdot)` is the discriminator model, :math:`x` is the
reference image, the reconstruction :math:`\hat x=\inverse{z}` for unconditional models (where :math:`z` are random latents) and
:math:`\hat x=\inverse{y,z}` for conditional models,
:math:`q(\cdot)` is a quality function (e.g :math:`q(x)=x` for WGAN) which can be set via :class:`deepinv.loss.adversarial.DiscriminatorMetric`.
Training alternates between generator :math:`\inverse{\cdot}` and discriminator
:math:`D` in a minimax game. When there are no ground truths (i.e.
unsupervised), this may be defined on the measurements :math:`y`
instead.

These examples can also be easily extended to train more complicated GANs such as CycleGAN.

"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize

import deepinv as dinv
from deepinv.loss import adversarial
from deepinv.utils import get_data_home
from deepinv.physics.generator import MotionBlurGenerator

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurments"
ORGINAL_DATA_DIR = get_data_home() / "Urban100"


# %%
# Generate dataset
# ~~~~~~~~~~~~~~~~
# In this example we use the Urban100 dataset resized to 128x128. We apply random
# motion blur physics using
# :class:`deepinv.physics.generator.MotionBlurGenerator`, and save the data
# using :func:`deepinv.datasets.generate_dataset`.
#

physics = dinv.physics.Blur(padding="circular", device=device)
blur_generator = MotionBlurGenerator((11, 11), device=device)

dataset = dinv.datasets.Urban100HR(
    root=ORGINAL_DATA_DIR,
    download=True,
    transform=Compose([ToTensor(), Resize(256), CenterCrop(128)]),
)

train_dataset, test_dataset = random_split(dataset, (0.8, 0.2))

# Generate data pairs x,y offline using a physics generator
dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    physics_generator=blur_generator,
    device=device,
    save_dir=DATA_DIR,
    batch_size=1,
)

train_dataloader = DataLoader(
    dinv.datasets.HDF5Dataset(
        dataset_path, train=True, load_physics_generator_params=True
    ),
    shuffle=True,
)
test_dataloader = DataLoader(
    dinv.datasets.HDF5Dataset(
        dataset_path, train=False, load_physics_generator_params=True
    ),
    shuffle=False,
)


# %%
# Define models
# ~~~~~~~~~~~~~
#
# We first define reconstruction network (i.e conditional generator) and
# discriminator network to use for adversarial training. For demonstration
# we use a simple U-Net as the reconstruction network and the
# discriminator from PatchGAN :footcite:p:`isola2017image`, but
# these can be replaced with any architecture e.g transformers, unrolled
# etc. Further discriminator models are in :ref:`adversarial models <adversarial>`.
#
# To train the discriminator, we also must provide an optimizer for it.
#


def get_models(model=None, D=None, lr_g=1e-4, lr_d=1e-4, device=device):
    if model is None:
        model = dinv.models.UNet(
            in_channels=3,
            out_channels=3,
            scales=2,
            circular_padding=True,
            batch_norm=False,
        ).to(device)

    if D is None:
        D = dinv.models.PatchGANDiscriminator(n_layers=2, batch_norm=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_g, weight_decay=1e-8)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_d, weight_decay=1e-8)

    return model, D, optimizer, optimizer_D


# %%
# Conditional GAN training
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Conditional GANs :footcite:p:`kupyn2018deblurgan` are a type of GAN where the generator is conditioned on a label or input.
# In the context of imaging, this can be used to generate images from a given measurement.
# In this example, we use a simple U-Net as the generator
# and a PatchGAN discriminator. The forward pass of the generator is given by:
#
# **Conditional GAN** forward pass:
#
# .. math:: \hat x = \inverse{y}
#
# **Conditional GAN** loss:
#
# .. math:: \mathcal{L}=\mathcal{L}_\text{sup}(\hat x, x)+\mathcal{L}_\text{adv}(\hat x, x;D)
#
# where :math:`\mathcal{L}_\text{sup}` is a supervised loss such as
# pixel-wise MSE or VGG Perceptual Loss.
#

model, D, optimizer, optimizer_D = get_models()


# %%
# We next define pixel-wise and adversarial losses as defined above. We use the
# MSE for the supervised pixel-wise metric for simplicity but this can be
# easily replaced with a perceptual loss if desired.
#
# .. hint::
#     To change the flavour of the GAN to WGAN, vanilla etc. pass in `metric` to `DiscriminatorMetric`.
#
# .. note::
#     The discriminator `D` is trained inside the loss by passing in `optimizer_D`. You can load a pretrained
#     discriminator using `loss.load_model`, and you can freeze the discriminator by omitting `optimizer_D`.
#

metric_gan = adversarial.DiscriminatorMetric(device=device)

loss = [
    dinv.loss.SupLoss(metric=torch.nn.MSELoss()),
    adversarial.SupAdversarialLoss(
        weight_adv=0.1,
        D=D,
        optimizer_D=optimizer_D,
        metric_gan=metric_gan,
        device=device,
    ),
]


# %%
# We are now ready to train the networks using the usual :class:`deepinv.Trainer`.
#
# .. warning::
#     GAN training is notoriously challenging and unstable, and training a well-trained GAN is beyond the scope of this example.
#     We demonstrate training 1 epoch for speed. Increase the number of epochs, the size/diversity of the dataset, and tune the
#     learning rates to produce better results.
#

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    train_dataloader=train_dataloader,
    epochs=1,
    losses=loss,
    optimizer=optimizer,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
)

model = trainer.train()

# %%
# Test the trained model and plot the results.
#

trainer.plot_images = True
trainer.test(test_dataloader)


# %%
# UAIR training
# ~~~~~~~~~~~~~
#
# Unsupervised Adversarial Image Reconstruction (UAIR) :footcite:p:`pajot2019unsupervised`
# is a method for solving inverse problems using generative models. In this
# example, we use a simple U-Net as the generator and discriminator, and
# train using the adversarial loss. The forward pass of the generator is defined as:
#
# **UAIR** forward pass:
#
# .. math:: \hat x = \inverse{y},
#
# **UAIR** loss:
#
# .. math:: \mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lambda\lVert \forw{\inverse{\hat y}}- \hat y\rVert^2_2,\quad\hat y=\forw{\hat x}.
#
# where :math:`\lambda` is a hyperparameter. We load the models and construct losses as defined above.

model, D, optimizer, optimizer_D = get_models(lr_g=1e-4, lr_d=4e-4)

loss = adversarial.UAIRLoss(
    D=D, optimizer_D=optimizer_D, physics_generator=blur_generator, device=device
)


# %%
# We are now ready to train the networks using the usual :class:`deepinv.Trainer`.
#
# .. warning::
#     GAN training is notoriously challenging and unstable, and training a well-trained GAN is beyond the scope of this example.
#     We demonstrate training 1 epoch for speed. Increase the number of epochs, the size/diversity of the dataset, and tune the
#     learning rates to produce better results.
#

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    train_dataloader=train_dataloader,
    epochs=1,
    losses=loss,
    optimizer=optimizer,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
)

model = trainer.train()

# %%
# Test the trained model and plot the results:
#

trainer.plot_images = True
trainer.test(test_dataloader)


# %%
# CSGM / AmbientGAN training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Compressed Sensing using Generative Models (CSGM) :footcite:p:`bora2017compressed` and AmbientGAN :footcite:p:`bora2018ambientgan` are two methods for solving inverse problems
# using generative models. CSGM uses a generative model to solve the inverse problem by optimising the latent
# space of the generator. AmbientGAN uses a generative model to solve the inverse problem by optimising the
# measurements themselves. Both methods are trained using an adversarial loss; the main difference is that CSGM requires
# a ground truth dataset (supervised loss), while AmbientGAN does not (unsupervised loss).
#
# In this example, we use a DCGAN as the
# generator and discriminator, and train using the adversarial loss. The forward pass of the generator is given by:
#
# **CSGM** forward pass at train time:
#
# .. math:: \hat x = \inverse{z},\quad z\sim \mathcal{N}(\mathbf{0},\mathbf{I}_k)
#
# **CSGM**/**AmbientGAN** forward pass at eval time:
#
# .. math:: \hat x = \inverse{\hat z}\quad\text{s.t.}\quad\hat z=\operatorname*{argmin}_z \lVert \forw{\inverse{z}}-y\rVert _2^2
#
# **CSGM** loss:
#
# .. math:: \mathcal{L}=\mathcal{L}_\text{adv}(\hat x, x;D)
#
# **AmbientGAN** loss (where :math:`\forw{\cdot}` is the physics):
#
# .. math:: \mathcal{L}=\mathcal{L}_\text{adv}(\forw{\hat x}, y;D)
#
# We next load the models and construct losses as defined above.

model = dinv.models.CSGMGenerator(
    dinv.models.DCGANGenerator(output_size=128, nz=100, ngf=32), inf_tol=1e-2
).to(device)
D = dinv.models.DCGANDiscriminator(ndf=32).to(device)

_, _, optimizer, optimizer_D = get_models(model=model, D=D, lr_g=2e-4, lr_d=2e-4)

# For AmbientGAN:
loss = adversarial.UnsupAdversarialLoss(D=D, optimizer_D=optimizer_D, device=device)

# For CSGM:
loss = adversarial.SupAdversarialLoss(D=D, optimizer_D=optimizer_D, device=device)


# %%
# As before, we can now train our models.
#
# .. warning::
#     GAN training is notoriously challenging and unstable, and training a well-trained GAN is beyond the scope of this example.
#     We demonstrate training 1 epoch for speed. Increase the number of epochs, the size/diversity of the dataset, and tune the
#     learning rates to produce better results.
#
# Note the train PSNR is meaningless as this generative model is trained on random latents.
#

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    train_dataloader=train_dataloader,
    epochs=1,
    losses=loss,
    optimizer=optimizer,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
)

model = trainer.train()


# %%
# Evaluate the generative model by running test-time optimisation
# using test measurements. Note that we do not get great results as CSGM /
# AmbientGAN relies on large datasets of diverse samples, and we run the
# optimisation to a relatively high tolerance for speed. Improve the results by
# running the optimisation for longer.
#

trainer.plot_images = True
trainer.test(test_dataloader)

# %%
# :References:
#
# .. footbibliography::
