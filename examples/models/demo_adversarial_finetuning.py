r"""
Fine-tuning models with adversarial losses
==========================================

This example shows you how to fine-tune reconstruction models using adversarial losses.

While the usual usage of adversarial losses is for training GANs (see :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py`),
they can also be used with a frozen pretrained discriminator to improve the perceptual performance of a pretrained
reconstruction model.
"""

import deepinv as dinv
import torch

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# %%
# We demonstrate fine-tuning the RAM foundation model :footcite:p:`terris2025reconstruct`,
# on a single demo image from CBSD68. The measurement operator is a 4x superresolution problem.

model = dinv.models.RAM(device="cpu")

x = dinv.utils.load_example("CBSD_0010.png", img_size=64, device=device)

physics = dinv.physics.Downsampling(
    (3, 64, 64), filter="bicubic", factor=4, device=device
)

y = physics(x)

# %%
# We use a pretrained discriminator provided in `Real-ESRGAN <https://github.com/xinntao/Real-ESRGAN>`_ :footcite:p:`wang2021realesrgan`,
# which has been trained on natural images for 2 or 4x superresolution problems, using
# a vanilla GAN loss.
#
# We use this to construct a loss consisting of a supervised pixel-wise loss (optionally pass in a perceptual metric if desired),
# and a supervised adversarial loss containing the frozen pretrained discriminator.
#

D = dinv.models.UNetDiscriminatorSN(pretrained_factor=4, device=device)

metric_gan = dinv.loss.adversarial.DiscriminatorMetric(
    metric=torch.nn.BCEWithLogitsLoss(),  # Vanilla GAN from BasicSR https://github.com/XPixelGroup/BasicSR
    device=device,
)

loss = [
    dinv.loss.SupLoss(metric=torch.nn.MSELoss()),
    dinv.loss.adversarial.SupAdversarialLoss(
        D=D, weight_adv=0.1, metric_gan=metric_gan, device=device
    ),
]

# %%
# We display performance on the base model, along with the
# discriminator feature maps showing how "real" the discriminator predicts
# the image as. Note that the reconstruction is classified as "more fake"
# than the ground truth, for now!
#
# We also calculate the distortion PSNR and the perceptual NIQE metrics
# for the images.

with torch.no_grad():
    x_net = model(y, physics)

psnr = dinv.metric.PSNR()
niqe = dinv.metric.NIQE(device=device)

dinv.utils.plot(
    {
        "GT": x,
        "Adjoint": physics.A_adjoint(y),
        "Base model recon": x_net,
        "GT discrim pred": D(x),
        "Recon discrim pred": D(x_net),
    },
    subtitles=[
        "",
        f"PSNR {psnr(physics.A_adjoint(y), x).item():.2f}",
        f"PSNR {psnr(x_net, x).item():.2f} NIQE {niqe(x_net, x).item():.2f}",
        f"D realness {D(x).mean().item():.1f}",
        f"D realness {D(x_net).mean().item():.1f}",
    ],
    fontsize=7,
)

# %%
# Perform fine-tuning on the model using the adversarial loss.
#
# .. note::
#     To keep the demo simple, we fine-tune on the test image using its ground truth. In practice, you should
#     fine-tune on either an external dataset of image pairs or use a self-supervised loss.
#     See :ref:`sphx_glr_auto_examples_models_demo_gan_imaging.py` for other adversarial losses available.

dataset = dinv.datasets.TensorDataset(x=x, y=y)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    train_dataloader=torch.utils.data.DataLoader(dataset),
    epochs=10,
    losses=loss,
    device="cpu",
    save_path=None,
)

model = trainer.train()

# %%
# Evaluate the fine-tuned model. Note how the "realness" of the reconstruction
# has increased, and both the distortion and perceptual metrics improve.

with torch.no_grad():
    x_net = model(y, physics)

dinv.utils.plot(
    {
        "GT": x,
        "Adjoint": physics.A_adjoint(y),
        "Base model recon": x_net,
        "GT discrim pred": D(x),
        "Recon discrim pred": D(x_net),
    },
    subtitles=[
        "",
        f"PSNR {psnr(physics.A_adjoint(y), x).item():.2f}",
        f"PSNR {psnr(x_net, x).item():.2f} NIQE {niqe(x_net, x).item():.2f}",
        f"D realness {D(x).mean().item():.1f}",
        f"D realness {D(x_net).mean().item():.1f}",
    ],
    fontsize=7,
)
