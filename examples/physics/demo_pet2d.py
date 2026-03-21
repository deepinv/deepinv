r"""
Positron emission tomography (PET) in 2D
========================================

This demo shows how to define a non time-of-flight PET scanner, simulate measurements
and reconstruct an image from them.

The PET forward model is defined as

.. math::

    y \sim \gamma \mathcal{P}(\frac{c \circ H(g*x) + s}{\gamma})

where :math:`H \in \mathbb{R}_{+}^{m \times n}` is the projection operator,
:math:`g \in \mathbb{R}_{+}^{n}` is a Gaussian blur kernel, :math:`x\in\mathbb{R}_{+}^{n}`
is the emission image, :math:`s \in \mathbb{R}_{+}^{m}` is the (expected) background,
:math:`\mathcal{P}` denotes Poisson noise with gain :math:`\gamma > 0`,
:math:`c=\exp(-H\mu)\in \mathbb{R}_{+}^{m}` is an (optional) attenuation term
with :math:`\mu \in \mathbb{R}_{+}^{n}` an attenuation map (typically obtained through an auxiliary CT scan).

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
# Here we define each pixel to have size :math:`3\times 3` mm
# such that the total area to reconstruct is of size :math:`38.4\times 38.4` cm
# which fits approximately a slice of a human chest.
#
# We use a PET scanner with a single ring of detectors, which is a polygon of
# 32 sides, with each side containing 16 detectors. This gives us a total of 32*16=512 detectors in total.
#
# .. tip::
#
#       You can play with different geometries and voxel sizes to get a good grasp of
#       the scanner geometry.
#
# .. note::
#
#      In this example, we normalize the forward operator :math:`\|A\|=1` (`normalize=True`) and
#      the Poisson counts to be between 0 and 1 (`normalize_counts=True`), to simplify reconstruction.
#      If you want to use the operator with real PET measurements, you will need to
#      carefully handle the normalization.

device = "cuda" if torch.cuda.is_available() else "cpu"
img_size = (128, 128)
voxel_size = (3, 3)

# number of sides of the polygone approximating a circle
num_sides = 32

# number of detectors per polygone side
num_lor_endpoints_per_side = 16

# choose a single ring for 2D
num_rings = 1

scanner = parallelproj.pet_scanners.DemoPETScannerGeometry(
    torch_compat,
    dev=device,
    num_rings=num_rings,
    num_sides=num_sides,
    num_lor_endpoints_per_side=num_lor_endpoints_per_side,
)

# gain of the device (associated with amount of dose/acq. time)
# larger gain -> more poisson noise -> harder reconstruction
gain = 0.001

physics = PET(
    device=device,
    voxel_size=voxel_size,
    scanner=scanner,
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
# We define a 2D phantom and attenuation map, whose shape is the same as the phantom.
#
# In practice, the attenuation is typically obtained with an auxiliary CT scan of the patient.

x, attenuation = generate_pet_phantom(img_size, device=device)

dinv.utils.plot([x, attenuation], titles=["Emission image", "Attenuation image"])

# %%
# Simulating measurements
# -----------------------
# We can generate measurements
# The shape of measurements is approximately `(B, 1, N, N/2)` where
# `N=num_lor_endpoints_per_side*num_sides` is the number of detectors per ring.
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
    [physics.attenuation, y, y2], ["sino. atten.", "meas.", "corrected meas."]
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

dinv.utils.plot(sensitivities, ["sensitivities"])

# %%
# MLEM reconstruction
# -------------------
#
# We run the standard MLEM reconstruction algorithm
# to obtain a reconstructed emission volume.
# We compare MLEM with the least-squares reconstruction.

gain = physics.noise_model.gain
data_fidelity = dinv.optim.PoissonLikelihood(
    bkg=background / gain,
    gain=gain,
    denormalize=True,
)

stepsize = 1.0
x_mlem = torch.ones_like(x)
with torch.no_grad():
    for i in range(100):
        grad = data_fidelity.grad(x=x_mlem, y=y, physics=physics) / gain
        x_mlem = x_mlem - stepsize * (x_mlem + 1e-9) / sensitivities * grad
        x_mlem = torch.clamp(x_mlem, min=0.0, max=5.0)


dinv.utils.plot([x, x_mlem, x_dag], ["Ground truth", "MLEM rec.", "L2 pseudoinv."])


# %%
# What next?
# ------------
# Now that you master the basics of PET, you can go further by
#
# - Check out the :ref:`3D PET example <sphx_glr_auto_examples_physics_demo_pet3d.py>`.
# - Reconstructing PET with learning-based methods (:ref:`PnP <iterative>`, :ref:`diffusion <sampling>`, :ref:`unrolled <unfolded>`, etc.)
# - Playing with the scanner setup: changing number of detectors, voxel size, etc.
