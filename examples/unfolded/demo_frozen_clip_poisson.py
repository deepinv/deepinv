r"""
Self-supervised Poisson demosaicing with a frozen CLIP prior
=============================================================

This example reconstructs photon-limited color images with the method of :footcite:t:`diazdelgado2026frozen`:
an unrolled plug-and-play ADMM whose prior is :class:`deepinv.models.FrozenCLIPDenoiser`,
i.e. multi-scale features of a frozen CLIP RN50 encoder mapped back to the image by a light trainable decoder.
Only the decoder is learned, from noisy measurements only, with the generalized R2R loss :footcite:p:`monroy2025generalized`
and equivariant imaging :footcite:p:`chen2021equivariant`.
"""

# sphinx_gallery_tags = ["Demosaicing", "Low-photon imaging"]

import torch
from torch.utils.data import DataLoader

import deepinv as dinv

torch.manual_seed(0)
device = dinv.utils.get_device()

# %%
# Poisson demosaicing
# -------------------
# We observe a Bayer (BGGR) mosaic corrupted by Poisson noise, :math:`y = \gamma \mathcal{P}(Ax / \gamma)` with gain :math:`\gamma = 0.01`.

x = dinv.utils.load_example("butterfly.png", img_size=256).to(device)
noise_model = dinv.physics.PoissonNoise(gain=0.01, clip_positive=True)
physics = dinv.physics.Demosaicing(
    img_size=x.shape[1:], noise_model=noise_model, device=device
)
y = physics(x)

# %%
# Unrolled ADMM with a frozen CLIP prior
# --------------------------------------
# We unroll :math:`T=2` iterations of :class:`deepinv.optim.ADMM` initialized at :math:`A^{\top}y`.
# Despite the Poisson noise, the data-fidelity is the :math:`\ell_2` surrogate :math:`\frac{1}{2}\|Ax-y\|^2`,
# whose proximal step is in closed form: an element-wise division for demosaicing.
# The prior step applies the pretrained denoiser, with weights shared across iterations.

denoiser = dinv.models.FrozenCLIPDenoiser(pretrained="download", device=device)
model = dinv.optim.ADMM(
    data_fidelity=dinv.optim.L2(),
    prior=dinv.optim.PnP(denoiser),
    stepsize=7.0,
    max_iter=2,
    unfold=True,
    trainable_params=[],
)

# %%
# The model was trained with GR2R, so as in the paper we evaluate it through :class:`deepinv.loss.R2RLoss`,
# which averages the reconstructions of several re-corrupted measurements.

r2r = dinv.loss.R2RLoss(noise_model=noise_model, alpha=0.2)
model = r2r.adapt_model(model).eval()

with torch.no_grad():
    x_hat = model(y, physics)

psnr = dinv.metric.PSNR()
x_adj = physics.A_adjoint(y)
dinv.utils.plot(
    {"Ground truth": x, r"$A^{\top}y$": x_adj, "Frozen CLIP prior": x_hat},
    subtitles=[
        "PSNR:",
        f"{psnr(x_adj, x).item():.2f} dB",
        f"{psnr(x_hat, x).item():.2f} dB",
    ],
)

# %%
# Self-supervised training
# ------------------------
# The decoder is trained without ground truth by minimizing
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
