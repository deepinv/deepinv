r"""
Imaging inverse problems with adversarial networks
==================================================

This example shows you how to train various networks using adversarial
training for deblurring problems. We demonstrate running training and
inference using a conditional GAN (i.e DeblurGAN), CSGM, AmbientGAN and
UAIR implemented in the ``deepinv`` library, and how to simply train
your own GAN by using ``deepinv.training.AdversarialTrainer``. These
examples can also be easily extended to train more complicated GANs such
as CycleGAN.

-  Kupyn et al., `DeblurGAN: Blind Motion Deblurring Using Conditional
   Adversarial
   Networks <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.pdf>`__
-  Bora et al., `Compressed Sensing using Generative
   Models <https://arxiv.org/abs/1703.03208>`__ (CSGM)
-  Bora et al., `AmbientGAN: Generative models from lossy
   measurements <https://openreview.net/forum?id=Hy7fDog0b>`__
-  Pajot et al., `Unsupervised Adversarial Image
   Reconstruction <https://openreview.net/forum?id=BJg4Z3RqF7>`__

Adversarial networks are characterised by the addition of an adversarial
loss :math:`\mathcal{L}_\text{adv}` to the standard reconstruction loss:

.. math:: \mathcal{L}_\text{adv}(x,\hat x;D)=\mathbb{E}_{x\sim p_x}\left[q(D(x))\right]+\mathbb{E}_{\hat x\sim p_{\hat x}}\left[q(1-D(\hat x))\right]

where :math:`D(\cdot)` is the discriminator model, :math:`x` is the
reference image, :math:`\hat x` is the estimated reconstruction,
:math:`q(\cdot)` is a quality function (e.g :math:`q(x)=x` for WGAN).
Training alternates between generator :math:`f` and discriminator
:math:`D` in a minimax game. When there are no ground truths (i.e
unsupervised), this may be defined on the measurements :math:`y`
instead.

**Conditional GAN** forward pass:

.. math:: \hat x = f(y)

**Conditional GAN** loss:

.. math:: \mathcal{L}=\mathcal{L}_\text{sup}(\hat x, x)+\mathcal{L}_\text{adv}(\hat x, x;D)

where :math:`\mathcal{L}_\text{sup}` is a supervised loss such as
pixel-wise MSE or VGG Perceptual Loss.

**CSGM**/**AmbientGAN** forward pass:

.. math:: \hat x = f(z),\quad z\sim \mathcal{N}(\mathbf{0},\mathbf{I}_k)

**CSGM** loss:

.. math:: \mathcal{L}=\mathcal{L}_\text{adv}(\hat x, x;D)

**AmbientGAN** loss (where :math:`A(\cdot)` is the physics):

.. math:: \mathcal{L}=\mathcal{L}_\text{adv}(A(\hat x), y;D)

**CSGM**/**AmbientGAN** forward pass at eval time:

.. math:: \hat x = f(\hat z)\quad\text{s.t.}\quad\hat z=\operatorname*{argmin}_z \lVert A(f(z))-y\rVert _2^2

**UAIR** forward pass:

.. math:: \hat x = f(y)

**UAIR** loss:

.. math:: \mathcal{L}=\mathcal{L}_\text{adv}(\hat y, y;D)+\lVert A(f(\hat y))- \hat y\rVert^2_2,\quad\hat y=A(\hat x)

"""

import deepinv as dinv
from deepinv.loss import adversarial
from deepinv.physics.generator import MotionBlurGenerator
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from torchvision.datasets.utils import download_and_extract_archive

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# %%
# Load data and apply some forward degradation to the images. For this
# example we use the Urban100 dataset resized to 128x128. We apply random
# motion blur physics using
# ``deepinv.physics.generator.MotionBlurGenerator``.
#

physics = dinv.physics.Blur(padding="circular")
blur_generator = MotionBlurGenerator((11, 11))

download_and_extract_archive(
    "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz?download=true",
    "Urban100",
    filename="Urban100_HR.tar.gz",
    md5="65d9d84a34b72c6f7ca1e26a12df1e4c",
)

train_dataset, test_dataset = random_split(
    ImageFolder(
        "Urban100", transform=Compose([ToTensor(), Resize(256), CenterCrop(128)])
    ),
    (0.8, 0.2),
)

train_dataloader = DataLoader(train_dataset, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=False)


# %%
# Define reconstruction network (i.e conditional generator) and
# discriminator network to use for adversarial training. For demonstration
# we use a simple U-Net as the reconstruction network and the
# discriminator from `PatchGAN <https://arxiv.org/abs/1611.07004>`__, but
# these can be replaced with any architecture e.g transformers, unrolled
# etc. Further discriminator models are in ``deepinv.models.gan``.
#


def get_models(model=None, D=None, lr_g=1e-4, lr_d=1e-4):
    if model is None:
        model = dinv.models.UNet(
            in_channels=3,
            out_channels=3,
            scales=2,
            circular_padding=True,
            batch_norm=False,
        )

    if D is None:
        D = dinv.models.PatchGANDiscriminator(n_layers=2, batch_norm=False)

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

model, D, optimizer, scheduler = get_models()


# %%
# Construct pixel-wise and adversarial losses as defined above. We use the
# MSE for the supervised pixel-wise metric for simplicity but this can be
# easily replaced with a perceptual loss if desired.
#

loss_g = [
    dinv.loss.SupLoss(metric=torch.nn.MSELoss()),
    adversarial.SupAdversarialGeneratorLoss(device=device),
]
loss_d = adversarial.SupAdversarialDiscriminatorLoss(device=device)


# %%
# Train the networks using ``AdversarialTrainer``. We only train for 3
# epochs for speed, but below we also show results with a pretrained model
# trained in the exact same way after 50 epochs.
#

model = dinv.training.AdversarialTrainer(
    model=model,
    D=D,
    physics=physics,
    physics_generator=blur_generator,
    online_measurements=True,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=3,
    losses=loss_g,
    losses_d=loss_d,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
).train()


# %%
# Show pretrained model results:
#

ckpt = torch.hub.load_state_dict_from_url(
    dinv.models.utils.get_weights_url("deblurgan-demo", "model.pth"),
    map_location=lambda s, _: s,
    file_name="model.pth",
)

model.load_state_dict(ckpt["state_dict"])

x, _ = next(iter(test_dataloader))
y = physics(x, **blur_generator.step())
dinv.utils.plot([x, y, model(y)], titles=["GT", "Measurement", "Reconstruction"])


# %%
# UAIR training
# ~~~~~~~~~~~~~
#

model, D, optimizer, scheduler = get_models(
    lr_g=1e-4, lr_d=4e-4
)  # learning rates from original paper


# %%
# Construct losses as defined above
#

loss_g = adversarial.UAIRGeneratorLoss(device=device)
loss_d = adversarial.UAIRDiscriminatorLoss(device=device)


# %%
# Train the networks using ``AdversarialTrainer``
#

model = dinv.training.AdversarialTrainer(
    model=model,
    D=D,
    physics=physics,
    physics_generator=blur_generator,
    online_measurements=True,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=3,
    losses=loss_g,
    losses_d=loss_d,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
).train()


# %%
# CSGM / AmbientGAN training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#

model = dinv.models.CSGMGenerator(
    dinv.models.DCGANGenerator(output_size=128, nz=100, ngf=32), inf_tol=1e-2
)
D = dinv.models.DCGANDiscriminator(ndf=32)
_, _, optimizer, scheduler = get_models(
    model=model, D=D, lr_g=2e-4, lr_d=2e-4
)  # learning rates from original paper


# %%
# Construct losses as defined above. We are free to choose between
# supervised and unsupervised adversarial losses, where supervised gives
# CSGM and unsupervised gives AmbientGAN.
#

loss_g = adversarial.SupAdversarialGeneratorLoss(device=device)
loss_d = adversarial.SupAdversarialDiscriminatorLoss(device=device)


# %%
# Train the networks using ``AdversarialTrainer``. Since inference is very
# slow for CSGM/AmbientGAN as it requires an optimisation, we only do one
# evaluation at the end. Note the train PSNR is meaningless as this
# generative model is trained on random latents.
#

trainer = dinv.training.AdversarialTrainer(
    model=model,
    D=D,
    physics=physics,
    physics_generator=blur_generator,
    online_measurements=True,
    train_dataloader=train_dataloader,
    epochs=3,
    losses=loss_g,
    losses_d=loss_d,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
)
model = trainer.train()


# %%
# Run evaluation of generative model by running test-time optimisation
# using test measurements. Note that we do not get great results as CSGM /
# AmbientGAN relies on large datasets of diverse samples, and we run the
# optimisation to a relatively high tolerance for speed.
#

psnr = trainer.test(test_dataloader)[0]
print("Test PSNR", psnr)
