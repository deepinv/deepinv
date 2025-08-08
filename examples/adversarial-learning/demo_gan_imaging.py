r"""
Imaging inverse problems with adversarial networks
==================================================

This example shows you how to train various networks using adversarial
training for deblurring problems. We demonstrate running training and
inference using a conditional GAN (i.e. DeblurGAN), CSGM, AmbientGAN and
UAIR implemented in the library, and how to simply train
your own GAN by using :class:`deepinv.training.AdversarialTrainer`. These
examples can also be easily extended to train more complicated GANs such
as CycleGAN.

This example is based on the papers DeblurGAN :footcite:p:`kupyn2018deblurgan`,
Compressed Sensing using Generative Models (CSGM) :footcite:p:`bora2017compressed`,
AmbiantGAN :footcite:p:`bora2018ambientgan`, and Unsupervised Adversarial Image Reconstruction (UAIR) :footcite:p:`pajot2019unsupervised`.

Adversarial networks are characterized by the addition of an adversarial
loss :math:`\mathcal{L}_\text{adv}` to the standard reconstruction loss:

.. math:: \mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]

where :math:`D(\cdot)` is the discriminator model, :math:`x` is the
reference image, :math:`\hat x` is the estimated reconstruction,
:math:`q(\cdot)` is a quality function (e.g :math:`q(x)=x` for WGAN).
Training alternates between generator :math:`G` and discriminator
:math:`D` in a minimax game. When there are no ground truths (i.e.
unsupervised), this may be defined on the measurements :math:`y`
instead.

"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize

import deepinv as dinv
from deepinv.loss import adversarial
from deepinv.utils.demo import get_data_home
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

    optimizer = dinv.training.adversarial.AdversarialOptimizer(
        torch.optim.Adam(model.parameters(), lr=lr_g, weight_decay=1e-8),
        torch.optim.Adam(D.parameters(), lr=lr_d, weight_decay=1e-8),
    )
    scheduler = dinv.training.adversarial.AdversarialScheduler(
        torch.optim.lr_scheduler.StepLR(optimizer.G, step_size=5, gamma=0.9),
        torch.optim.lr_scheduler.StepLR(optimizer.D, step_size=5, gamma=0.9),
    )

    return model, D, optimizer, scheduler


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
# .. math:: \hat x = G(y)
#
# **Conditional GAN** loss:
#
# .. math:: \mathcal{L}=\mathcal{L}_\text{sup}(\hat x, x)+\mathcal{L}_\text{adv}(\hat x, x;D)
#
# where :math:`\mathcal{L}_\text{sup}` is a supervised loss such as
# pixel-wise MSE or VGG Perceptual Loss.
#

G, D, optimizer, scheduler = get_models()


# %%
# We next define pixel-wise and adversarial losses as defined above. We use the
# MSE for the supervised pixel-wise metric for simplicity but this can be
# easily replaced with a perceptual loss if desired.
#

loss_g = [
    dinv.loss.SupLoss(metric=torch.nn.MSELoss()),
    adversarial.SupAdversarialGeneratorLoss(device=device),
]
loss_d = adversarial.SupAdversarialDiscriminatorLoss(device=device)


# %%
# We are now ready to train the networks using :class:`deepinv.training.AdversarialTrainer`.
# We load the pretrained models that were trained in the exact same way after 50 epochs,
# and fine-tune the model for 1 epoch for a quick demo.
# You can find the pretrained models on HuggingFace https://huggingface.co/deepinv/adversarial-demo.
# To train from scratch, simply comment out the model loading code and increase the number of epochs.
#

ckpt = torch.hub.load_state_dict_from_url(
    dinv.models.utils.get_weights_url("adversarial-demo", "deblurgan_model.pth"),
    map_location=lambda s, _: s,
)

G.load_state_dict(ckpt["state_dict"])
D.load_state_dict(ckpt["state_dict_D"])
optimizer.load_state_dict(ckpt["optimizer"])

trainer = dinv.training.AdversarialTrainer(
    model=G,
    D=D,
    physics=physics,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=1,
    losses=loss_g,
    losses_d=loss_d,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
)

G = trainer.train()

# %%
# Test the trained model and plot the results. We compare to the pseudo-inverse as a baseline.
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
# .. math:: \hat x = G(y),
#
# **UAIR** loss:
#
# .. math:: \mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lVert \forw{\inverse{\hat y}}- \hat y\rVert^2_2,\quad\hat y=\forw{\hat x}.
#
# We next load the models and construct losses as defined above.

G, D, optimizer, scheduler = get_models(
    lr_g=1e-4, lr_d=4e-4
)  # learning rates from original paper

loss_g = adversarial.UAIRGeneratorLoss(device=device)
loss_d = adversarial.UnsupAdversarialDiscriminatorLoss(device=device)


# %%
# We are now ready to train the networks using :class:`deepinv.training.AdversarialTrainer`.
# Like above, we load a pretrained model trained in the exact same way for 50 epochs,
# and fine-tune here for a quick demo with 1 epoch.
#

ckpt = torch.hub.load_state_dict_from_url(
    dinv.models.utils.get_weights_url("adversarial-demo", "uair_model.pth"),
    map_location=lambda s, _: s,
)

G.load_state_dict(ckpt["state_dict"])
D.load_state_dict(ckpt["state_dict_D"])
optimizer.load_state_dict(ckpt["optimizer"])

trainer = dinv.training.AdversarialTrainer(
    model=G,
    D=D,
    physics=physics,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=1,
    losses=loss_g,
    losses_d=loss_d,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
)
G = trainer.train()

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

G = dinv.models.CSGMGenerator(
    dinv.models.DCGANGenerator(output_size=128, nz=100, ngf=32), inf_tol=1e-2
).to(device)
D = dinv.models.DCGANDiscriminator(ndf=32).to(device)
_, _, optimizer, scheduler = get_models(
    model=G, D=D, lr_g=2e-4, lr_d=2e-4
)  # learning rates from original paper

# For AmbientGAN:
loss_g = adversarial.UnsupAdversarialGeneratorLoss(device=device)
loss_d = adversarial.UnsupAdversarialDiscriminatorLoss(device=device)

# For CSGM:
loss_g = adversarial.SupAdversarialGeneratorLoss(device=device)
loss_d = adversarial.SupAdversarialDiscriminatorLoss(device=device)


# %%
# As before, we can now train our models. Since inference is very
# slow for CSGM/AmbientGAN as it requires an optimisation, we only do one
# evaluation at the end. Note the train PSNR is meaningless as this
# generative model is trained on random latents.
# Like above, we load a pretrained model trained in the exact same way for 50 epochs,
# and fine-tune here for a quick demo with 1 epoch.
#

ckpt = torch.hub.load_state_dict_from_url(
    dinv.models.utils.get_weights_url("adversarial-demo", "csgm_model.pth"),
    map_location=lambda s, _: s,
)

G.load_state_dict(ckpt["state_dict"])
D.load_state_dict(ckpt["state_dict_D"])
optimizer.load_state_dict(ckpt["optimizer"])

trainer = dinv.training.AdversarialTrainer(
    model=G,
    D=D,
    physics=physics,
    train_dataloader=train_dataloader,
    epochs=1,
    losses=loss_g,
    losses_d=loss_d,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
)
G = trainer.train()


# %%
# Eventually, we run evaluation of the generative model by running test-time optimisation
# using test measurements. Note that we do not get great results as CSGM /
# AmbientGAN relies on large datasets of diverse samples, and we run the
# optimisation to a relatively high tolerance for speed. Improve the results by
# running the optimisation for longer.
#

trainer.test(test_dataloader)

# %%
# :References:
#
# .. footbibliography::
