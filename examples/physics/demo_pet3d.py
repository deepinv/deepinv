r"""
Positron emission tomography (PET) in 3D
========================================

This demo shows how to define a non time-of-flight PET scanner, simulate measurements
and reconstruct a volume from them.

The (unnormalized) PET forward model is defined as

.. math::

    y \sim \gamma \mathcal{P}(c \circ H(g*x) + s)

where :math:`H \in \mathbb{R}_{+}^{m \times n}` is the projection operator,
:math:`g \in \mathbb{R}_{+}^{n}` is a Gaussian blur kernel, :math:`x\in\mathbb{R}_{+}^{n}`
is the emission image, :math:`s \in \mathbb{R}_{+}^{m}` is the (expected) background,
:math:`\mathcal{P}` denotes Poisson noise,
:math:`c=\exp(-H\mu)\in \mathbb{R}_{+}^{m}` is an (optional) attenuation term
with :math:`\mu \in \mathbb{R}_{+}^{n}` an attenuation map (typically obtained through an auxiliary CT scan).

.. note::

    In this example, we consider the unnormalized case, which allows to obtain quantitative reconstructions (i.e., :math:`x` has real
    physical units). The operator also can be used in a normalized setting (forcing :math:`\|A\|_2=1` and normalizing counts to be between 0 and 1).
    See also the :ref:`normalized 2D PET example <sphx_glr_auto_examples_physics_demo_pet2d.py>`.
    When using deep learning-based reconstruction methods, is often easier to consider the normalized case, but a special attention is required
    to denormalize the reconstructions and obtain physical units.

.. tip::

    If you prefer to get started with PET on a simpler 2D problem, please check out :ref:`the 2D PET demo <sphx_glr_auto_examples_physics_demo_pet2d.py>`.

.. note::

    PET physics relies on the `parallelproj` library.
    Please `download it <https://parallelproj.readthedocs.io/en/stable/installation.html>`_ to run this example.

"""

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

# number of sides of the polygone approximating a circle
num_sides = 32

# number of detectors per polygone side
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


physics = PET(
    device=device,
    voxel_size=voxel_size,
    scanner=scanner,
    fwhm_data_mm=fwhm_data_mm,
    img_size=img_size,
    normalize_counts=False,
    normalize=False,
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

# longer acquisition times -> more counts -> easier reconstruction
acquisition_time_factor = 10.0
x = x * acquisition_time_factor

dinv.utils.plot(
    [x[..., mid_slice], attenuation[..., mid_slice]],
    titles=["Emission image", "Attenuation image"],
)

# %%
# Simulating measurements
# -----------------------
# We can generate measurements
# The shape of measurements is approximately `(B, 1, N, N/2, R^2)` where
# `N=num_lor_endpoints_per_side*num_sides` is the number of detectors per ring
# and `R` is the number of rings.
# This provides one measurement for every possible Line of Response (LOR), or in other words 'rays', connecting
# two detectors in the scanner.
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
#   but it should be provided in image space, i.e., :math:`\mu`, to the physics.

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
# The sensitivity map is defined as the back-projection of a sinogram of ones, which corresponds to the number of rays intersecting each voxel.
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
# We run the standard MLEM reconstruction algorithm
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
#
# We compare MLEM with the least-squares reconstruction.

data_fidelity = dinv.optim.PoissonLikelihood(
    bkg=background,
    denormalize=False,
)

x_mlem = torch.ones_like(x)
with torch.no_grad():
    for i in range(50):
        grad = data_fidelity.grad(x=x_mlem, y=y, physics=physics)
        preconditioner = (x_mlem + 1e-9) / (sensitivities + 1e-9)
        x_mlem = x_mlem - preconditioner * grad
        x_mlem = torch.clamp(x_mlem, min=0.0)

dinv.utils.plot(
    [x[..., mid_slice], x_mlem[..., mid_slice], x_dag[..., mid_slice]],
    ["Ground truth", "MLEM rec.", "L2 pseudoinv."],
    cbar=True,
)


# %%
# What next?
# ------------
# Now that you master the basics of PET, you can go further by
#
# - Reconstructing PET with learning-based methods (:ref:`PnP <iterative>`, :ref:`diffusion <sampling>`, :ref:`unrolled <unfolded>`, etc.)
# - Playing with the scanner setup: changing number of detectors, voxel size, etc.
