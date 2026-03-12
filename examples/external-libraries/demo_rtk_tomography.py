"""
3D CBCT volume reconstruction using the RTK backend and Total-Variation (TV) prior
========================================================================================

This example shows 3D cone-beam CT (CBCT) reconstruction using the Reconstruction Toolkit (RTK) library.

We generate a Shepp–Logan phantom, simulate its projections using the RTK CUDA projector operator, and add noise to the measurements. The reconstruction is performed using a Proximal Gradient Descent (PGD) algorithm with a Total Variation (TV) prior, and the result is compared against the Feldkamp–Davis–Kress (FDK) reconstruction.

We do this with the :class:`deepinv.physics.TomographyWithRTK` operator, which wraps the RTK <https://docs.openrtk.org/>_ library. This class is defined at this end of this example.

This example requires ITK and RTK to be installed in your current environment. They can be installed using: `pip install itk-rtk-cuda128`

"""

import itk
from itk import RTK as rtk
import torch
import deepinv as dinv
from pathlib import Path

from deepinv.optim.data_fidelity import L2
from deepinv.optim import PGD
from deepinv.utils.plotting import plot, plot_curves
from deepinv.physics import GaussianNoise
from deepinv.physics import TomographyWithRTK

import matplotlib.pyplot as plt

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#
BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"


# %%
# Generation of the Shepp-Logan phantom
# -----------------------------------------
# Here, we generate the 3D phantom through the RTK library and convert it from an ITK image to a Torch tensor.
# Note that the phantom values are from 0 to 2 rather than the normalized 0 to 1.

ImageType = itk.Image[itk.F, 3]
slice_of_interest = 24  # Use for visualization

source = rtk.constant_image_source(
    size=[64, 64, 64],
    spacing=[4.0] * 3,
    ttype=[ImageType],
    origin=[-126.0, -126.0, -126.0],
)

shepploganFilter = rtk.DrawSheppLoganFilter.New()
shepploganFilter.SetInput(source)
shepploganFilter.Update()

# Convert to pytorch tensor
itk_gpu_image = itk.cuda_image_from_image(shepploganFilter.GetOutput())
itk_gpu_tensor = torch.tensor(itk_gpu_image).unsqueeze(0).unsqueeze(0)

# Display slice of the volume
plt.imshow(
    (itk_gpu_tensor.cpu()[0, 0, :, slice_of_interest, :]),
    vmin=0.99,
    vmax=1.04,
    cmap="gray",
    origin="lower",
)
plt.colorbar()
plt.show()

# %%
# Definition of forward operator and noise model
# -----------------------------------------------
# First we define the geometry and then the projections and volume source coordinate informations.
# For the noise, we assume it follows a gaussian distribution.
#

# Setup the geometry
numberOfProjections = 600
angularArc = 360.0
sid = 300
sdd = 500
geometry = rtk.ThreeDCircularProjectionGeometry.New()
for i in range(0, numberOfProjections):
    angle = i * angularArc / numberOfProjections
    geometry.AddProjection(sid, sdd, angle)

projection_stack_information = {
    "spacing": [1, 1, 1],
    "size": [100, 100, 600],
    "origin": [-49.5, -49.5, 0.0],
}


volume_information = {
    "spacing": [1, 1, 1],
    "size": [64, 64, 64],
    "origin": [-31.5, -31.5, -31.5],
}

# Instantiation of the operator
noise_level = 3e-1
physics = TomographyWithRTK(
    geometry=geometry,
    projection_stack_information=projection_stack_information,
    volume_information=volume_information,
    verbose=True,
    normalize=False,
    noise_model=GaussianNoise(sigma=noise_level),
    mode="conebeam",
    ray_step_size=1.0,
)

# Application of the operator and computation of the pseudo inverse using the FDK algorithm
observation = physics(itk_gpu_tensor)
fdk = physics.fbp(observation)

# %%
# Set up the optimization algorithm to solve the inverse problem.
# --------------------------------------------------------------------------------
# The problem we want to minimize is the following:
#
# .. math::
#
#     \begin{equation*}
#     \underset{x}{\operatorname{min}} \,\, \frac{1}{2} \|Ax-y\|_2^2 + \lambda \|Dx\|_{1,2}(x),
#     \end{equation*}
#
#
# where :math:`\frac{1}{2} \|A(x)-y\|_2^2` is the data-fidelity term, :math:`\lambda \|Dx\|_{2,1}(x)` is the total variation (TV)
# norm of the image :math:`x`, and :math:`\lambda>0` is a regularisation parameter.
#
# We use a Proximal Gradient Descent (PGD) algorithm to solve the inverse problem.


# Select the data fidelity term
data_fidelity = L2()
prior = dinv.optim.prior.TVPrior(n_it_max=20)

# Logging parameters
verbose = True
plot_convergence_metrics = (
    True  # compute performance and convergence metrics along the algorithm.
)

# Algorithm parameters
print("Calculating the operator norm this may take some time...")
scaling = (
    1
    / physics.compute_sqnorm(
        torch.randn_like(itk_gpu_tensor), max_iter=100, tol=1e0
    ).item()
)
print(scaling)

stepsize = 0.99 * scaling
lamb = 20  # TV regularisation parameter
max_iter = 1000
early_stop = True

# Instantiate the algorithm class to solve the problem.
model = PGD(
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=verbose,
    stepsize=stepsize,
    lambda_reg=lamb,
    custom_init=lambda observation, physics: scaling
    * physics.A_adjoint(observation),  # initialize with backprojection
)


# %%
# Evaluate the model and plot the results
# --------------------------------------------------------------------
#
# The model returns the output and the metrics computed along the iterations.
# The PSNR is computed w.r.t the ground truth image in ``test_imgs``.

# run the model on the problem.
x_model, metrics = model(
    observation, physics, x_gt=itk_gpu_tensor, compute_metrics=True
)  # reconstruction with PGD algorithm

# compute PSNR
print(
    f"Filtered Back-Projection PSNR: {dinv.metric.PSNR(max_pixel=itk_gpu_tensor.max().item())(itk_gpu_tensor.unsqueeze(0), fdk).item():.2f} dB"
)
print(
    f"PGD reconstruction PSNR: {dinv.metric.PSNR(max_pixel=itk_gpu_tensor.max())(itk_gpu_tensor.unsqueeze(0), x_model).item():.2f} dB"
)

imgs = [
    itk_gpu_tensor[0, :, :, slice_of_interest, :],
    fdk[0, :, :, slice_of_interest, :],
    x_model[0, :, :, slice_of_interest, :],
]
plot(
    imgs,
    titles=["GT", "Filtered Back-Projection", "Recons."],
    save_dir=RESULTS_DIR,
    vmin=0.99,
    vmax=1.04,
    rescale_mode="clip",
    origin="lower",
)

# plot convergence curves
if plot_convergence_metrics:
    plot_curves(metrics, save_dir=RESULTS_DIR)

plot(
    observation[0, :, 50, :, :],
    titles=[f"Noisy sinogram"],
    vmax=55,
    save_dir=RESULTS_DIR,
    rescale_mode="clip",
)
