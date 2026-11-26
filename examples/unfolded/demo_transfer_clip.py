r"""
Transfer CLIP denoiser and Frozen CLIP Priors for Poisson demosaicing
=====================================================================

This example shows two uses of the Transfer CLIP denoiser :class:`deepinv.models.TransferCLIP` :footcite:p:`cheng2024transfer`,
which maps the multi-scale features of a frozen CLIP RN50 encoder back to the image with a light trainable decoder:

1. Denoising with the original weights, which were trained on Gaussian noise but also remove unseen noise.
2. Poisson demosaicing with Frozen CLIP Priors :footcite:p:`diazdelgado2026frozen`, an unrolled plug-and-play ADMM whose prior is this denoiser.
   Its decoder is trained from noisy measurements only, with the generalized R2R loss :footcite:p:`monroy2025generalized`
   and equivariant imaging :footcite:p:`chen2021equivariant`.

The original code of Transfer CLIP is available at https://github.com/alwaysuu/CLIPDenoising.
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
# The default weights of :footcite:t:`cheng2024transfer` were trained on Gaussian noise with standard deviation :math:`\sigma = 15/255`.
# Thanks to the frozen CLIP features, the denoiser also removes noise that it was not trained on, such as Poisson noise.

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
# We now observe a Bayer (BGGR) mosaic with :class:`deepinv.physics.Demosaicing`, corrupted by :class:`deepinv.physics.PoissonNoise`:
# :math:`y = \gamma \mathcal{P}(Ax / \gamma)` with gain :math:`\gamma = 0.01`.

noise_model = dinv.physics.PoissonNoise(gain=0.01, clip_positive=True)
physics = dinv.physics.Demosaicing(
    img_size=x.shape[1:], noise_model=noise_model, device=device
)
y = physics(x)

# %%
# Frozen CLIP Priors
# ------------------
# Frozen CLIP Priors :footcite:p:`diazdelgado2026frozen` unrolls :math:`T=2` iterations of :class:`deepinv.optim.ADMM` initialized at :math:`A^{\top}y`.
# Despite the Poisson noise, the data fidelity is the :math:`\ell_2` surrogate :math:`\frac{1}{2}\|Ax-y\|^2`,
# whose proximal step is in closed form: an element-wise division for demosaicing.
# The prior step applies the Transfer CLIP denoiser, with weights shared across iterations.
# We load the Frozen CLIP Priors weights, which were trained for this task from the Transfer CLIP weights.
# The original code of Frozen CLIP Priors is available at https://github.com/LauraCD2/frozen-clip-priors.


def unrolled_admm(denoiser):
    return dinv.optim.ADMM(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.PnP(denoiser),
        stepsize=7.0,
        max_iter=2,
        unfold=True,
        trainable_params=[],
    )


denoiser = dinv.models.TransferCLIP(
    pretrained="poisson_demosaicing", aug_level=0.02, device=device
)
model = unrolled_admm(denoiser)

# %%
# The model was trained with GR2R, so we evaluate it through :class:`deepinv.loss.R2RLoss`,
# which averages the reconstructions of several re-corrupted measurements.

r2r = dinv.loss.R2RLoss(noise_model=noise_model, alpha=0.2)
model = r2r.adapt_model(model).eval()

with torch.no_grad():
    x_hat = model(y, physics)

x_adj = physics.A_adjoint(y)
dinv.utils.plot(
    {"Ground truth": x, r"$A^{\top}y$": x_adj, "Frozen CLIP Priors": x_hat},
    subtitles=[
        "PSNR:",
        f"{psnr(x_adj, x).item():.2f} dB",
        f"{psnr(x_hat, x).item():.2f} dB",
    ],
)

# %%
# Self-supervised training under strong Poisson noise
# ---------------------------------------------------
# We now increase the noise to a gain :math:`\gamma = 0.05`, the strongest level in :footcite:t:`diazdelgado2026frozen`,
# and train Frozen CLIP Priors as in the paper. The unrolled model starts from the original Transfer CLIP weights,
# and only the decoder of the denoiser is trained, without ground truth, by minimizing
#
# .. math::
#     \mathcal{L} = \mathcal{L}_{\text{GR2R}} + 0.1\, \mathcal{L}_{\text{EI}},
#
# where :class:`deepinv.loss.R2RLoss` compares a reconstruction from a re-corrupted measurement with an independent re-corruption,
# and :class:`deepinv.loss.EILoss` enforces equivariance to rotations.
#
# .. tip::
#     For Poisson deblurring, the paper uses a 9x9 Gaussian blur :class:`deepinv.physics.BlurFFT` with ``filter=dinv.physics.functional.gaussian_blur(sigma=(1, 1))``,
#     whose proximal step is computed with FFTs, together with :math:`T=3`, ``stepsize=1.0`` and :class:`deepinv.transform.Scale` in the EI loss.

noise_model = dinv.physics.PoissonNoise(gain=0.05, clip_positive=True)
physics = dinv.physics.Demosaicing(
    img_size=x.shape[1:], noise_model=noise_model, device=device
)
y = physics(x)

denoiser = dinv.models.TransferCLIP(
    pretrained="download", aug_level=0.02, device=device
)
model = unrolled_admm(denoiser)

with torch.no_grad():
    x_init = model(y, physics)

# %%
# We train for 50 steps on 128x128 measurements of five other images, which takes less than a minute on CPU.
# Training longer on a larger dataset such as :class:`deepinv.datasets.BSDS500`, as in the paper, further improves the results.

x_train = torch.cat(
    [
        dinv.utils.load_example(name, img_size=128)
        for name in (
            "leaves.png",
            "celeba_example.jpg",
            "barbara.jpeg",
            "69037.png",
            "div2k_valid_hr_0877.png",
        )
    ]
).to(device)
physics_train = dinv.physics.Demosaicing(
    img_size=(128, 128), noise_model=noise_model, device=device
)
train_dataset = dinv.datasets.TensorDataset(y=physics_train(x_train))

r2r = dinv.loss.R2RLoss(noise_model=noise_model, alpha=0.2)
model = r2r.adapt_model(model)
losses = [
    r2r,
    dinv.loss.EILoss(transform=dinv.transform.Rotate(), weight=0.1, no_grad=True),
]

trainer = dinv.Trainer(
    model=model,
    physics=physics_train,
    train_dataloader=DataLoader(train_dataset, batch_size=1, shuffle=True),
    optimizer=torch.optim.Adam(denoiser.parameters(), lr=1e-4),
    epochs=10,
    losses=losses,
    metrics=None,
    device=device,
    save_path=None,
    show_progress_bar=False,
)
model = trainer.train().eval()

with torch.no_grad():
    x_ft = model(y, physics)

x_adj = physics.A_adjoint(y)
dinv.utils.plot(
    {
        "Ground truth": x,
        r"$A^{\top}y$": x_adj,
        "Frozen CLIP Priors\n(before training)": x_init,
        "Frozen CLIP Priors\n(after training)": x_ft,
    },
    subtitles=[
        "PSNR:",
        f"{psnr(x_adj, x).item():.2f} dB",
        f"{psnr(x_init, x).item():.2f} dB",
        f"{psnr(x_ft, x).item():.2f} dB",
    ],
    figsize=(10, 3),
)

# %%
# :References:
#
# .. footbibliography::
