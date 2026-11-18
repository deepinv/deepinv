r"""
Transfer CLIP denoiser and unrolled Poisson demosaicing
=======================================================

This example shows two uses of :class:`deepinv.models.TransferCLIP`, the denoiser of :footcite:t:`cheng2024transfer`,
which maps the multi-scale features of a frozen CLIP RN50 encoder back to the image with a light trainable decoder:

1. denoising with the original weights, trained on Gaussian noise, including noise not seen during training;
2. Poisson demosaicing with the method of :footcite:t:`diazdelgado2026frozen`, an unrolled plug-and-play ADMM whose prior is this denoiser.
   Its decoder is learned from noisy measurements only, with the generalized R2R loss :footcite:p:`monroy2025generalized`
   and equivariant imaging :footcite:p:`chen2021equivariant`.
"""

# sphinx_gallery_tags = ["Denoising", "Demosaicing", "Low-photon imaging"]

import torch
from torch.utils.data import DataLoader

import deepinv as dinv

torch.manual_seed(0)
device = dinv.utils.get_device()

# %%
# Denoising with Transfer CLIP
# ----------------------------
# The pretrained weights of :footcite:t:`cheng2024transfer` were trained on Gaussian noise with standard deviation 15/255.
# Thanks to the frozen CLIP features, the denoiser also removes noise it was not trained on, here Poisson noise.

x = dinv.utils.load_example("butterfly.png", img_size=256).to(device)
denoiser = dinv.models.TransferCLIP(pretrained="download", device=device)

y_gauss = dinv.physics.GaussianNoise(sigma=15 / 255)(x)
y_poisson = dinv.physics.PoissonNoise(gain=0.01)(x)
with torch.no_grad():
    x_gauss, x_poisson = denoiser(y_gauss), denoiser(y_poisson)

psnr = dinv.metric.PSNR()
images = [y_gauss, x_gauss, y_poisson, x_poisson]
dinv.utils.plot(
    images,
    titles=["Gaussian noise", "Denoised", "Poisson noise", "Denoised"],
    subtitles=[f"{psnr(im, x).item():.2f} dB" for im in images],
)

# %%
# Poisson demosaicing
# -------------------
# We now observe a Bayer (BGGR) mosaic corrupted by Poisson noise, :math:`y = \gamma \mathcal{P}(Ax / \gamma)` with gain :math:`\gamma = 0.01`.

noise_model = dinv.physics.PoissonNoise(gain=0.01, clip_positive=True)
physics = dinv.physics.Demosaicing(
    img_size=x.shape[1:], noise_model=noise_model, device=device
)
y = physics(x)

# %%
# Unrolled ADMM with a Transfer CLIP prior
# ----------------------------------------
# Following :footcite:t:`diazdelgado2026frozen`, we unroll :math:`T=2` iterations of :class:`deepinv.optim.ADMM` initialized at :math:`A^{\top}y`.
# Despite the Poisson noise, the data-fidelity is the :math:`\ell_2` surrogate :math:`\frac{1}{2}\|Ax-y\|^2`,
# whose proximal step is in closed form: an element-wise division for demosaicing.
# The prior step applies the Transfer CLIP denoiser, with weights shared across iterations.
# We load the weights of :footcite:t:`diazdelgado2026frozen`, fine-tuned from the previous ones for this task.

denoiser = dinv.models.TransferCLIP(
    pretrained="poisson_demosaicing", aug_level=0.02, device=device
)
model = dinv.optim.ADMM(
    data_fidelity=dinv.optim.L2(),
    prior=dinv.optim.PnP(denoiser),
    stepsize=7.0,
    max_iter=2,
    unfold=True,
    trainable_params=[],
)

# %%
# The model was trained with GR2R, so we evaluate it through :class:`deepinv.loss.R2RLoss`,
# which averages the reconstructions of several re-corrupted measurements.

r2r = dinv.loss.R2RLoss(noise_model=noise_model, alpha=0.2)
model = r2r.adapt_model(model).eval()

with torch.no_grad():
    x_hat = model(y, physics)

x_adj = physics.A_adjoint(y)
dinv.utils.plot(
    {"Ground truth": x, r"$A^{\top}y$": x_adj, "Unrolled Transfer CLIP": x_hat},
    subtitles=[
        "PSNR:",
        f"{psnr(x_adj, x).item():.2f} dB",
        f"{psnr(x_hat, x).item():.2f} dB",
    ],
)

# %%
# Self-supervised training
# ------------------------
# Only the decoder of the denoiser is trained, without ground truth, by minimizing
#
# .. math::
#     \mathcal{L} = \mathcal{L}_{\text{GR2R}} + 0.1\, \mathcal{L}_{\text{EI}},
#
# where :class:`deepinv.loss.R2RLoss` compares a reconstruction from a re-corrupted measurement with an independent re-corruption,
# and :class:`deepinv.loss.EILoss` enforces equivariance to rotations.
# Here we only fine-tune the pretrained model for one epoch on 128x128 measurements of three other images.
#
# .. tip::
#     For Poisson deblurring, use :class:`deepinv.physics.BlurFFT` with ``filter=dinv.physics.functional.gaussian_blur(sigma=(1, 1))`` (9x9),
#     whose data-fidelity step is computed with FFTs, :math:`T=3`, ``stepsize=1.0``, and :class:`deepinv.transform.Scale` in the EI loss.

x_train = torch.cat(
    [
        dinv.utils.load_example(name, img_size=128)
        for name in ("leaves.png", "celeba_example.jpg", "barbara.jpeg")
    ]
).to(device)
physics_train = dinv.physics.Demosaicing(
    img_size=(128, 128), noise_model=noise_model, device=device
)
train_dataset = dinv.datasets.TensorDataset(y=physics_train(x_train))

losses = [r2r, dinv.loss.EILoss(transform=dinv.transform.Rotate(), weight=0.1)]

trainer = dinv.Trainer(
    model=model,
    physics=physics_train,
    train_dataloader=DataLoader(train_dataset, batch_size=1, shuffle=True),
    optimizer=torch.optim.Adam(denoiser.parameters(), lr=1e-5),
    epochs=1,
    losses=losses,
    metrics=None,
    device=device,
    save_path=None,
    show_progress_bar=False,
)
model = trainer.train().eval()

with torch.no_grad():
    x_ft = model(y, physics)

dinv.utils.plot(
    {"Ground truth": x, "Pretrained": x_hat, "Fine-tuned": x_ft},
    subtitles=[
        "PSNR:",
        f"{psnr(x_hat, x).item():.2f} dB",
        f"{psnr(x_ft, x).item():.2f} dB",
    ],
)

# %%
# :References:
#
# .. footbibliography::
