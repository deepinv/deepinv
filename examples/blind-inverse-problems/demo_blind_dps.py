r"""
Blind deblurring with parallel diffusion models
===============================================

Recover a sharp image and its unknown blur kernel with :class:`deepinv.sampling.BlindDPS`
from :footcite:t:`chung2023parallel`. Two diffusion priors are guided by the same
measurement residual through both denoisers. We compare with kernel identification
followed by :class:`deepinv.optim.DPIR`, as in
:ref:`sphx_glr_auto_examples_blind-inverse-problems_demo_blind_deblurring.py`.

This example uses the authors' `pretrained models and reference implementation
<https://github.com/BlindDPS/blind-dps>`_. DeepInverse integrates the continuous reverse
SDE with an Euler solver; it uses the predicted noise, without the checkpoints'
learned DDPM variance. Sampling is computationally expensive, especially on CPU.
"""

# %%
import torch
import deepinv as dinv

device = dinv.utils.get_device()
torch.manual_seed(123)

# %%
# Generate a blurred face
# -----------------------
# The image prior was trained on 256 x 256 FFHQ faces. We use a simple isotropic
# Gaussian blur with standard deviation 3 pixels, as in Section 4.1 of the paper.
# The 64 x 64 support matches the pretrained kernel prior. The ground-truth
# kernel is used only to simulate the measurement; reconstruction receives a
# blur operator with no filter. Reflected boundaries avoid wrap-around blur.

x = dinv.utils.load_example(
    "celeba_example.jpg", img_size=256, resize_mode="resize", device=device
)
kernel_size = 64
kernel = dinv.physics.functional.gaussian_blur(
    sigma=3.0, psf_size=(kernel_size, kernel_size), device=device
)
# The authors add noise with sigma=0.02 to images in [-1, 1], equivalent to
# sigma=0.01 in DeepInverse's [0, 1] convention.
sigma = 0.02 / 2
measurement_physics = dinv.physics.Blur(
    filter=kernel,
    padding="reflect",
    use_fft=True,
    noise_model=dinv.physics.GaussianNoise(sigma),
    device=device,
)
y = measurement_physics(x)

# %%
# Load the image and kernel priors
# --------------------------------
# Download the two original checkpoints once (about 422 MiB total). Local paths
# can be passed to ``DiffUNet(pretrained=...)`` instead. The grayscale kernel model
# has two output channels: predicted noise and learned variance.

weights_dir = dinv.utils.get_cache_home() / "models" / "blind_dps"
weights_dir.mkdir(parents=True, exist_ok=True)
for name, file_id in (
    ("ffhq_10m.pt", "1nAhgjU8C6DCkOLmWTuPIzA6PMNkNmE5Z"),
    ("kernel_checkpoint.pt", "11Xn8tsisCCIrv3aFyitmj55Sc13Wwb8j"),
):
    path = weights_dir / name
    if not path.exists():
        torch.hub.download_url_to_file(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            str(path),
        )

denoiser = (
    dinv.models.DiffUNet(pretrained=str(weights_dir / "ffhq_10m.pt"), dropout=0.0)
    .to(device)
    .eval()
    .requires_grad_(False)
)
kernel_denoiser = (
    dinv.models.DiffUNet(
        in_channels=1,
        out_channels=2,
        img_size=64,
        model_channels=64,
        channel_mult=(1, 2, 3, 4),
        dropout=0.0,
        pretrained=str(weights_dir / "kernel_checkpoint.pt"),
    )
    .to(device)
    .eval()
    .requires_grad_(False)
)

# %%
# Choose the sampling budget
# ---------------------------
# Start with 200 solver time points for a quick test, using about five times
# fewer denoiser evaluations. Set ``num_steps = 1000`` for the paper's sampling
# budget. Shorter runs can reduce reconstruction quality. After changing this
# value, rerun the next cell; it also adjusts the guidance scaling.

num_steps = 200

# %%
# Jointly estimate the image and kernel
# -------------------------------------
# Both denoisers follow DeepInverse's [0, 1] convention. Internally the diffusion
# states use [-1, 1]. Kernels are normalized before evaluating the residual.
# The authors use 1000 diffusion steps and a guidance step size of 0.3 for both
# priors. Here the Euler update multiplies guidance by beta(t) * dt, so we divide
# by this factor to keep the same per-step correction. The factor 2 accounts for
# evaluating the residual in [0, 1] instead of [-1, 1]. This matches the guidance
# scale, while the prior updates still use the continuous SDE described above.
# We use the diffusion kernel prior without the optional sparsity penalty.

guidance_step = 0.3
dt = (1 - 0.001) / (num_steps - 1)
sde = dinv.sampling.VariancePreservingDiffusion(alpha=1.0, device=device)


def guidance_weight(t):
    return 2 * guidance_step / (sde.forward_diffusion(t) ** 2 * dt)


physics = dinv.physics.Blur(padding="reflect", use_fft=True, device=device)
model = dinv.sampling.BlindDPS(
    denoiser,
    kernel_denoiser,
    kernel_size=kernel_size,
    num_steps=num_steps,
    weight=guidance_weight,
    kernel_weight=guidance_weight,
    sde=sde,
    device=device,
    verbose=True,
)

x_hat, kernel_hat = model(y, physics, seed=123)

# %%
# Compare with kernel identification and DPIR
# -------------------------------------------
# This baseline estimates spatially varying kernels from the same measurement,
# then performs non-blind deblurring. It also receives no ground-truth kernel.
# Its circular convolution assumes matching opposite boundaries. We apply
# :func:`deepinv.physics.functional.liu_jia_pad` :footcite:p:`liu2008reducing`
# before kernel estimation and deconvolution, then crop away the padded margin.
# BlindDPS uses reflected convolution directly on the original 256 x 256 image.

margin = kernel_size // 2
y_padded = dinv.physics.functional.liu_jia_pad(y, padding=(margin, margin))
kernel_estimator = dinv.models.KernelIdentificationNetwork(device=device)
baseline_physics = dinv.physics.SpaceVaryingBlur(
    padding="circular", mask_first=False, device=device
)
with torch.no_grad():
    baseline_physics.update(**kernel_estimator(y_padded))
    x_dpir = dinv.optim.DPIR(sigma=sigma, device=device)(
        y_padded, baseline_physics
    ).clamp(0, 1)
    x_dpir = x_dpir[..., margin:-margin, margin:-margin]

psnr = dinv.metric.PSNR()
dinv.utils.plot(
    {"Ground truth": x, "Blurred": y, "Kernel ID + DPIR": x_dpir, "BlindDPS": x_hat},
    subtitles=[
        "",
        f"PSNR: {psnr(y, x).item():.2f} dB",
        f"PSNR: {psnr(x_dpir, x).item():.2f} dB",
        f"PSNR: {psnr(x_hat, x).item():.2f} dB",
    ],
    figsize=(16, 4),
)
dinv.utils.plot({"True kernel": kernel, "BlindDPS kernel": kernel_hat})

# %%
# :References:
#
# .. footbibliography::
