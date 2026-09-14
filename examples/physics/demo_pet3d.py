r"""
Positron emission tomography (PET) in 3D
========================================

This demo shows how to define a non time-of-flight PET scanner, simulate measurements
and reconstruct a volume from them.

The (unnormalized) PET forward model is defined as

.. math::

    y \sim \gamma \mathcal{P}(c \circ H(g*x) + b)

where :math:`H \in \mathbb{R}_{+}^{m \times n}` is the projection operator,
:math:`g \in \mathbb{R}_{+}^{n}` is a Gaussian blur kernel, :math:`x\in\mathbb{R}_{+}^{n}`
is the emission image, :math:`b \in \mathbb{R}_{+}^{m}` is the (expected) background,
:math:`\mathcal{P}` denotes Poisson noise,
:math:`c=\exp(-H\mu)\in \mathbb{R}_{+}^{m}` is an (optional) attenuation term
with :math:`\mu \in \mathbb{R}_{+}^{n}` an attenuation map (typically obtained through an auxiliary CT scan).

.. note::

    In this example, we consider the unnormalized case, which allows to obtain quantitative reconstructions (i.e., :math:`x` has real
    physical units). The operator also can be used in a normalized setting (forcing :math:`\|A\|_2=1` and normalizing counts to be between 0 and 1).
    See also the :ref:`normalized 2D PET example <sphx_glr_auto_examples_physics_demo_pet2d.py>`.
    When using deep learning-based reconstruction methods, it is often easier to consider the normalized case, but a special attention is required
    to denormalize the reconstructions and obtain physical units.

.. tip::

    If you prefer to get started with PET on a simpler 2D problem, please check out :ref:`the 2D PET demo <sphx_glr_auto_examples_physics_demo_pet2d.py>`.

.. note::

    This operator requires the `parallelproj` package to be installed.
    This in turn requires :ref:`installing deepinv via pixi or conda <install>`,
    but not pypi/uv (as `parallelproj` is not currently available on pypi).

    If you are working on a conda environment, you can install `parallelproj` as

    ::

        conda install -c conda-forge parallelproj


    If you are working on a pixi installation, simply do

    ::

        pixi install -e full

    which installs all optional dependencies.

    Check the `parallelproj` documentation for more details: https://parallelproj.readthedocs.io/en/stable/.

"""

# %%
import time

import matplotlib.pyplot as plt
import deepinv as dinv
from deepinv.physics import PET
from deepinv.utils.phantoms import generate_pet_phantom
import torch
import parallelproj
from array_api_compat import torch as torch_compat

# %%
# Setup a minimal non-TOF PET projector
# -------------------------------------
#
# Here we define each voxel to have size :math:`3\times 3\times 3` mm
# such that the total volume to reconstruct is of size :math:`38.4\times 38.4\times 7.2` cm
# which fits approximately a portion of a human chest.
#
# The maximum achievable resolution (in high count settings) is typically proportional to the full-width at half
# maximum (FWHM) of the Gaussian blur kernel, which here is set to 4 mm.
#
# We use a PET scanner with 8 rings of detectors, each ring being a polygon of
# 32 sides, and each side containing 16 detectors. This gives us a total of 32*16=512 detectors per ring.
#
# .. tip::
#
#       You can play with different geometries and voxel sizes to get a good grasp of
#       the scanner geometry.
#

device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = (128, 128, 24)
voxel_size = (3, 3, 3)

# number of sides of the polygon approximating a circle
num_sides = 32

# number of detectors per polygon side
num_lor_endpoints_per_side = 16

# number of rings of detectors on the depth axes
num_rings = 8

scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_rings=num_rings,
    num_sides=num_sides,
    num_lor_endpoints_per_side=num_lor_endpoints_per_side,
)

# FWHM of the Gaussian blur kernel in mm
fwhm_data_mm = 4

# gain of the device:
# higher gains are associated to lower dose and/or shorter acquisition times,
# while lower gains are associated to higher dose and/or longer acquisition times.
# larger gain -> more poisson noise -> harder reconstruction
gain = 0.001

physics = PET(
    device=device,
    voxel_size=voxel_size,
    scanner=scanner,
    fwhm_data_mm=fwhm_data_mm,
    img_size=img_size,
    normalize_counts=True,
    normalize=True,
    gain=gain,
)

physics.plot_geometry()

# %%
# Define a phantom and attenuation map
# ------------------------------------
#
# We define a 3D phantom and attenuation map, whose shape is the same as the phantom.
#
# In practice, the attenuation is typically obtained with an auxiliary CT scan of the patient.

x, attenuation = generate_pet_phantom(img_size, device=device)
mid_slice = img_size[-1] // 2

dinv.utils.plot(
    [x[..., mid_slice], attenuation[..., mid_slice]],
    titles=["Emission image", "Attenuation image"],
)

# %%
# Simulating measurements
# -----------------------
# The shape of measurements is approximately `(B, 1, N, N/2, R^2)` where
# `N=num_lor_endpoints_per_side*num_sides` is the number of detectors per ring
# and `R` is the number of rings.
# This provides one measurement for every possible Line of Response (LOR), or in other words 'rays', connecting
# two detectors in the scanner, which are arranged in a sinogram format, with the first axis
# corresponding to the angle of the ray, the second axis corresponding to the distance of the ray to the center of the field of view
# and the last axis corresponding to the depth of the ray (i.e., which rings of detectors are connected by the ray)
#
# .. tip::
#
#     The size of measurements is independent of the chosen `img_size`

y = physics(x)

print(
    f"Measurements shape={tuple(y.shape)}, range=({y.min().item():.2f},{y.max().item():.2f})"
)

# %%
# Setting up background and attenuation
# -------------------------------------
# The attenuation term reduces the amount of signal measured in rays that
# go through highly attenuating regions, such as bones. This makes the reconstruction more challenging, but also more realistic.
#
# In PET, we generally have access to a realization of the background,
# i.e., :math:`\tilde{s} \sim \mathcal{P}(s)`, which is a Poisson random variable with mean :math:`s`.
#
# Both attenuation and background are stored as "physics parameters" which are patient dependent
# and can be updated via :meth:`physics.update(...) <deepinv.physics.Physics.update>` or by passing them as kwargs in
# :meth:`physics(x, ...) <deepinv.physics.Physics.forward>`, :meth:`physics.A(x, ...) <deepinv.physics.Physics.A>` or
# :meth:`physics.A_adjoint(y, ...) <deepinv.physics.LinearPhysics.A_adjoint>`.
#
# .. note::
#
#   The attenuation is stored in the physics in sinogram space as :math:`\exp(-\mu)` to speed up computations,
#   but it can be provided either in image space, i.e., :math:`\mu`, to the physics, or in sinogram space, i.e., :math:`\exp(-\mu)`.
#   The class figures out the attenuation space by comparing it to `img_size`.

expected_background = torch.ones_like(y) * x.max() * 0.05
background = physics.generate_background(expected_background)
physics.update(attenuation=attenuation, background=background)
y = physics(x)
y2 = y - background
dinv.utils.plot(
    [physics.attenuation[..., mid_slice], y[..., mid_slice], y2[..., mid_slice]],
    ["sino. atten.", "meas.", "corrected meas."],
    figsize=(6, 6),
)

# %%
# Backprojection and sensitivities
# --------------------------------
# We backproject the data to visualize the sensitivity map of the scanner.
# The sensitivity map is defined as the back-projection of a sinogram of ones :math:`s = A^\top \mathbf{1}`, which corresponds to the number of rays intersecting each voxel.
#
# Here we also obtain a simple linear least-squares reconstruction by using
# :meth:`A_dagger <deepinv.physics.LinearPhysics.A_dagger>`.

with torch.no_grad():
    x_dag = physics.A_dagger(y - background)
    sensitivities = physics.A_adjoint(torch.ones_like(y))

print(f"Norm operator: {physics.compute_norm(x):.2f}")

dinv.utils.plot(sensitivities[..., mid_slice], ["sensitivities"])

# %%
# MLEM reconstruction
# -------------------
#
# We run the standard MLEM reconstruction algorithm :footcite:p:`sheppMaximumLikelihoodReconstruction1982`
# to obtain a reconstructed emission volume.
#
# The algorithm can be seen as a preconditioned gradient descent on the negative log-likelihood of the Poisson model:
#
# .. math::
#
#   x^{(k+1)} = x^{(k)} - P \nabla f(Ax^{(k)}+b,y)
#
# where :math:`f` is the Poisson data-fidelity term, :math:`P=\mathrm{diag}(\frac{x}{A^T\mathbf{1}})` is a preconditioner
# and :math:`b` is the background.

gain = physics.noise_model.gain
mlem_iter = 40


def _sync():
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


nrmse = dinv.metric.NRMSE()


# With ``denormalize=True``, the likelihood is evaluated in the count domain.
# The background produced by PET is in the normalized measurement domain, so it
# must be divided by the gain as well.
data_fidelity = dinv.optim.PoissonLikelihood(
    gain=gain,
    bkg=background / gain,
    denormalize=True,
)
model_mlem = dinv.optim.MLEM(
    data_fidelity=data_fidelity,
    prior=None,
    max_iter=mlem_iter,
    custom_metrics={
        "nrmse": lambda _values, _x_prev, x_cur: nrmse(x_cur.unsqueeze(0), x).item()
    },
)

with torch.no_grad():
    _sync()
    start = time.perf_counter()
    x_mlem, metrics_mlem = model_mlem(
        y, physics, init=torch.ones_like(x), compute_metrics=True
    )
    _sync()
    mlem_time = time.perf_counter() - start

print(f"MLEM runtime: {mlem_time:.2f} s for {mlem_iter} iterations")

psnr = dinv.metric.PSNR(max_pixel=x.max().item())

psnr_mlem = psnr(x_mlem, x)
psnr_dag = psnr(x_dag, x)
nrmse_mlem = nrmse(x_mlem, x)
nrmse_dag = nrmse(x_dag, x)

dinv.utils.plot(
    [x[..., mid_slice], x_mlem[..., mid_slice], x_dag[..., mid_slice]],
    ["Ground truth", f"MLEM ({mlem_iter} it.)", "L2 pseudoinv."],
    subtitles=[
        "Reference",
        f"PSNR: {psnr_mlem.item():.2f} dB\n" f"NRMSE: {100 * nrmse_mlem.item():.2f}%",
        f"PSNR: {psnr_dag.item():.2f} dB\n" f"NRMSE: {100 * nrmse_dag.item():.2f}%",
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    figsize=(8, 4),
    cbar=True,
)

# %%
# Accelerating MLEM with ordered subsets
# ---------------------------------------
#
# One MLEM iteration evaluates the forward and adjoint operators using the complete
# PET acquisition. This can be slow when the scanner geometry and its measurements
# are large, especially in 3D.
#
# Ordered-Subsets Expectation-Maximization (OSEM) :footcite:p:`hudsonAcceleratedImageReconstruction1994`
# partitions the measurements and the forward operator into :math:`L` matching angular subsets indexed by
# :math:`l=1,\ldots,L`:
#
# .. math::
#
#   y = (y_1,\ldots,y_L), \qquad
#   A = (A_1^\top,\ldots,A_L^\top)^\top.
#
# OSEM applies an EM update successively with each pair :math:`(y_l,A_l)`. Each subset
# update is cheaper than a full MLEM iteration and updates the complete image, so OSEM
# generally reaches a useful reconstruction in fewer passes over the complete acquisition.

# %%
# OSEM reconstruction
# -------------------
#
# OSEM accepts the full measurements and physics and splits them internally. Alternatively,
# pre-split inputs can be created with :func:`deepinv.physics.split_measurements` and
# :func:`deepinv.physics.split_physics` and passed directly to :class:`deepinv.optim.OSEM`.

osem_iter = 3
num_subsets = 16

model_osem = dinv.optim.OSEM(
    data_fidelity=data_fidelity,
    prior=None,
    max_iter=osem_iter,
    num_subsets=num_subsets,
    custom_metrics={
        "nrmse": lambda _values, _x_prev, x_cur: nrmse(x_cur.unsqueeze(0), x).item()
    },
)

with torch.no_grad():
    _sync()
    start = time.perf_counter()
    x_osem, metrics_osem = model_osem(
        y,
        physics,
        init=torch.ones_like(x),
        compute_metrics=True,
    )
    _sync()
    osem_time = time.perf_counter() - start

print(
    f"OSEM runtime: {osem_time:.2f} s "
    f"for {osem_iter} iterations "
    f"({num_subsets} subsets)"
)
print(f"Reconstruction speedup MLEM/OSEM: {mlem_time / osem_time:.2f}x")

psnr_osem = psnr(x_osem, x)
nrmse_osem = nrmse(x_osem, x)

dinv.utils.plot(
    [
        x[..., mid_slice],
        x_mlem[..., mid_slice],
        x_osem[..., mid_slice],
    ],
    [
        "Ground truth",
        f"MLEM ({mlem_iter} it.)",
        f"OSEM ({osem_iter} it.)",
    ],
    subtitles=[
        "Reference",
        f"PSNR: {psnr_mlem.item():.2f} dB\n" f"NRMSE: {100 * nrmse_mlem.item():.2f}%",
        f"PSNR: {psnr_osem.item():.2f} dB\n" f"NRMSE: {100 * nrmse_osem.item():.2f}%",
    ],
    rescale_mode="clip",
    vmin=0,
    vmax=x.max().item(),
    figsize=(8, 4),
    cbar=True,
)

# We also compare the Poisson objective and NRMSE after every iteration.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(metrics_mlem["cost"][0], label="MLEM")
axes[0].plot(metrics_osem["cost"][0], label="OSEM")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Poisson NLL")
axes[0].legend()

axes[1].plot([100 * v for v in metrics_mlem["nrmse"][0]], label="MLEM")
axes[1].plot([100 * v for v in metrics_osem["nrmse"][0]], label="OSEM")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("NRMSE (%)")
axes[1].yaxis.set_major_formatter("{x:.0f}%")
axes[1].legend()
fig.tight_layout()

# %%
# What next?
# ------------
# Now that you master the basics of PET, you can go further by
#
# - Reconstructing PET with learning-based methods (:ref:`PnP <iterative>`, :ref:`diffusion <sampling>`, :ref:`unrolled <unfolded>`, etc.)
# - Playing with the scanner setup: changing number of detectors, voxel size, etc.

# %%
# :References:
#
# .. footbibliography::
