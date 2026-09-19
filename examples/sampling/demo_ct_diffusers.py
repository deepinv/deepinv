r"""
Compare data consistency for a pretrained DM4CT diffusion model
===============================================================

This example compares the noisy data-fidelity approximations available in
DeepInv for sparse-view CT reconstruction with a pretrained pixel-space model
from the `DM4CT benchmark <https://github.com/DM4CT/DM4CT>`_. The checkpoint is
published as a Diffusers pipeline on `HuggingFace
<https://huggingface.co/jiayangshi/lodochallenge_pixel_diffusion>`_ and is loaded
with :class:`deepinv.models.DiffusersDenoiserWrapper`.

The DM4CT paper stresses that CT reconstruction depends critically on the
balance between the diffusion prior and data consistency. It finds that no one
conditioning method is uniformly best: pseudoinverse or optimization-based
conditioning generally enforces the measurements more strongly, while gradient
conditioning can preserve more prior detail but may hallucinate anatomy. We
therefore compare all six concrete :class:`deepinv.sampling.NoisyDataFidelity`
implementations under the same acquisition, sampler and random seed, alongside
filtered back-projection (FBP) and SIRT baselines.

We use the authors' ``L506_000.tif`` example and their 40-view medical CT
configuration. This is important because the checkpoint was trained using one
global ``[-1, 1]`` normalization of the Low Dose CT Grand Challenge data; an
unrelated CT image with a different intensity normalization causes distribution
shift before the inverse solver is considered.

.. warning::

    The checkpoint is approximately 455 MB and operates on 512 x 512 images.
    It is loaded in float16 while ASTRA and the sampler use float32. ALD,
    Score-SDE and ILVR do not differentiate through the UNet. DPS, PiGDM and
    Moment Matching do and require substantially more memory. Edit
    ``methods_to_run`` below to run a subset. This example requires ``diffusers``,
    ``transformers``, ``astra-toolbox`` and ``tifffile``.

.. note::

    DM4CT also publishes latent-diffusion checkpoints. Those models require
    decoder-aware measurement conditioning, whereas
    :class:`deepinv.models.DiffusersDenoiserWrapper` currently wraps only the
    pipeline UNet. This example is therefore restricted to pixel-space models.
"""

# %%
from time import perf_counter

import torch

import deepinv as dinv

# %% Configuration
# -----------------
if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA GPU.")

device = torch.device("cuda")
model_dtype = torch.float16
sampling_dtype = torch.float32  # required by ASTRA
img_size = 512
num_angles = 40
noise_std = 0.0
num_steps = 100
seed = 0

# Run the lower-memory methods first. If memory is limited, remove the final
# three denoiser-Jacobian methods from this list.
methods_to_run = [
    "ALD",
    "Score-SDE",
    "ILVR",
    "DPS",
    "PiGDM",
    "Moment Matching",
]

# %% Load a distribution-matched test slice
# ------------------------------------------
# This is the test image used in the DM4CT repository's reconstruction example.
# It is stored in the checkpoint's training range, ``[-1, 1]``. The Diffusers
# wrapper follows DeepInv's standard denoiser convention, so we map it to
# ``[0, 1]`` before defining the inverse problem.
image_url = (
    "https://raw.githubusercontent.com/DM4CT/DM4CT/" "main/lodochallenge/L506_000.tif"
)
x = dinv.utils.load_tiff(dinv.utils.load_url(image_url), dtype=sampling_dtype).to(
    device
)
x = (x + 1.0) / 2.0

# %% Simulate the sparse-view CT acquisition
# -------------------------------------------
# Forty noiseless views reproduce configuration (i) of the DM4CT paper. Set
# ``noise_std`` above to compare the methods under additive Gaussian noise. The
# same value is exposed through ``physics.noise_model`` to the methods which use
# the assumed measurement variance.
physics = dinv.physics.TomographyWithAstra(
    img_size=(img_size, img_size),
    angles=num_angles,
    normalize=True,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_std),
)
y = physics(x)

dinv.utils.plot(
    [x, y],
    titles=["Ground truth", "Sparse-view measurements"],
    figsize=(8, 4),
    rescale_mode=None,
)

# %% Evaluation helpers
# ---------------------
# PSNR and SSIM measure agreement with the reference. The relative data residual
# directly measures consistency with the acquired sinogram and helps reveal a
# plausible-looking but weakly conditioned sample.
psnr = dinv.metric.PSNR()
ssim = dinv.metric.SSIM()
reconstructions = {}
results = {}


def evaluate(name, reconstruction, elapsed, peak_memory):
    reconstruction = reconstruction.detach().clip(0.0, 1.0)
    reconstructions[name] = reconstruction
    results[name] = {
        "PSNR": psnr(reconstruction, x).item(),
        "SSIM": ssim(reconstruction, x).item(),
        "data fit": (
            torch.linalg.vector_norm(physics.A(reconstruction) - y)
            / torch.linalg.vector_norm(y)
        ).item(),
        "time": elapsed,
        "memory": peak_memory,
    }


def run_and_evaluate(name, reconstruction_fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = perf_counter()
    reconstruction = reconstruction_fn()
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    evaluate(name, reconstruction, elapsed, peak_memory)


# %% FBP and SIRT baselines
# -------------------------
run_and_evaluate("FBP", lambda: physics.A_dagger(y, fbp=True))

sirt = dinv.optim.SIRT(
    data_fidelity=dinv.optim.L2(),
    max_iter=200,
    early_stop=False,
    verbose=False,
    show_progress_bar=False,
)
run_and_evaluate("SIRT", lambda: sirt(y, physics))

# %% Load the pretrained pixel-space model
# -----------------------------------------
denoiser = dinv.models.DiffusersDenoiserWrapper(
    model_id="jiayangshi/lodochallenge_pixel_diffusion",
    dtype=model_dtype,
    device=device,
)

# DM4CT uses a 1,000-step linear DDPM schedule with beta_start=1e-4 and
# beta_end=2e-2. These correspond to beta_min=0.1 and beta_max=20 in the
# continuous VP parametrization. We use 100 steps for a practical comparison and
# keep the same SDE and Brownian path for every data fidelity.
solver = dinv.sampling.EulerSolver(
    t_start=1.0,
    t_end=1e-3,
    num_steps=num_steps,
    rng=torch.Generator(device=device),
)
sde = dinv.sampling.VariancePreservingDiffusion(
    denoiser=denoiser,
    beta_min=0.1,
    beta_max=20.0,
    alpha=1.0,
    solver=solver,
    minus_one_one=False,
    dtype=sampling_dtype,
    device=device,
)

# %% Compare all noisy data-fidelity approximations
# -------------------------------------------------
# The DM4CT paper tunes DPS to a step size of 10 for its medical dataset. We use
# that value with the original residual-norm guidance. The paper's PGDM setting
# maps to DeepInv's PiGDM implementation, for which we use its configuration-(i)
# value of 1. The remaining methods were not part of DM4CT, so their neutral
# default weight is retained. These values are starting points, not universally
# optimal hyperparameters.
data_fidelities = {
    "ALD": dinv.sampling.ALDDataFidelity(weight=1.0),
    "Score-SDE": dinv.sampling.ScoreSDEDataFidelity(
        weight=1.0,
        rng=torch.Generator(device=device).manual_seed(seed),
    ),
    "ILVR": dinv.sampling.ILVRDataFidelity(
        weight=1.0,
        rng=torch.Generator(device=device).manual_seed(seed),
    ),
    "DPS": dinv.sampling.DPSDataFidelity(
        denoiser=denoiser,
        weight=10.0,
        guidance="norm",
    ),
    "PiGDM": dinv.sampling.PiGDMDataFidelity(
        denoiser=denoiser,
        weight=1.0,
        clip=(0.0, 1.0),
        cg_max_iter=3,
        cg_tol=1e-4,
    ),
    "Moment Matching": dinv.sampling.MomentMatchingDataFidelity(
        denoiser=denoiser,
        weight=1.0,
        clip=(0.0, 1.0),
        cg_max_iter=3,
        cg_tol=1e-4,
    ),
}

for method_name in methods_to_run:
    reconstructor = dinv.sampling.PosteriorDiffusion(
        data_fidelity=data_fidelities[method_name],
        sde=sde,
        solver=solver,
        minus_one_one=False,
        dtype=sampling_dtype,
        device=device,
        verbose=True,
    )
    run_and_evaluate(
        method_name,
        lambda reconstructor=reconstructor: reconstructor(
            y=y,
            physics=physics,
            x_init=x.shape,
            seed=seed,
        ),
    )
    del reconstructor

# %% Quantitative comparison
# ---------------------------
print(
    f"{'Method':<18} {'PSNR':>8} {'SSIM':>8} "
    f"{'Rel. data fit':>14} {'Time (s)':>10} {'Peak GiB':>10}"
)
for name, metrics in results.items():
    print(
        f"{name:<18} {metrics['PSNR']:>8.2f} {metrics['SSIM']:>8.3f} "
        f"{metrics['data fit']:>14.3e} {metrics['time']:>10.1f} "
        f"{metrics['memory']:>10.2f}"
    )


def comparison_title(name):
    metrics = results[name]
    return f"{name}\n{metrics['PSNR']:.1f} dB, fit {metrics['data fit']:.1e}"


# %% Visual comparison
# ---------------------
# The first figure contains methods whose data-fidelity update does not require a
# UNet Jacobian. The second contains the more memory-intensive, denoiser-based
# approximations. All images use the same fixed ``[0, 1]`` display range.
first_group = ["FBP", "SIRT", "ALD", "Score-SDE", "ILVR"]
first_group = [name for name in first_group if name in reconstructions]
dinv.utils.plot(
    [x] + [reconstructions[name] for name in first_group],
    titles=["Ground truth"] + [comparison_title(name) for name in first_group],
    figsize=(3 * (len(first_group) + 1), 3),
    rescale_mode=None,
)

second_group = ["DPS", "PiGDM", "Moment Matching"]
second_group = [name for name in second_group if name in reconstructions]
if second_group:
    dinv.utils.plot(
        [x] + [reconstructions[name] for name in second_group],
        titles=["Ground truth"] + [comparison_title(name) for name in second_group],
        figsize=(3 * (len(second_group) + 1), 3),
        rescale_mode=None,
    )

# %%
