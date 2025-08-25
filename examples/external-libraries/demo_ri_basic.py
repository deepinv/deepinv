"""
Radio interferometric imaging with deepinverse
==============================================

In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse.
The following example and data are taken from :footcite:t:`aghabiglou2024r2d2`.
If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check `BASPLib <https://basp-group.github.io/BASPLib/>`_.
"""

# %%
# Import required packages
# ----------------------------------------------------------------------------------------
# We rely on the `TorchKbNufft` as the non-uniform FFT backend in this problem.
# This first snippet is just here to check that dependencies are installed properly.

import torch
import numpy as np
import torchkbnufft as tkbn

import deepinv as dinv
from deepinv.utils.plotting import plot, plot_curves, scatter_plot, plot_inset
from deepinv.utils.demo import load_np_url, get_image_url, get_degradation_url
from deepinv.utils.tensorlist import dirac_like

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# The RI measurement operator
# ----------------------------------------------------------------------------------------
# The RI inverse problem aims at restoring the target image :math:`x\in \mathbb{R}^{n}` from complex measurements (or visibilities) :math:`y \in \mathbb{C}^{m}`, reads:
#
# .. math::
#   \begin{equation*}
#       y = Ax+\epsilon,
#   \end{equation*}
#
# where :math:`A` can be decomposed as :math:`A = GFZ \in \mathbb{C}^{m \times n}`.
# There, :math:`G \in \mathbb{C}^{m \times d}` is a sparse interpolation matrix,
# encoding the non-uniform Fourier transform,
# :math:`F \in \mathbb{C}^{d\times d}` is the 2D Discrete Fourier Transform,
# :math:`Z \in \mathbb{R}^{d\times n}` is a zero-padding operator,
# incorporating the correction for the convolution performed through the operator :math:`G`,
# and :math:`\epsilon \in \mathbb{C}^{m}` is a realization of some i.i.d. Gaussian random noise.
#
# This operator can be implemented with `TorchKbNUFFT <https://github.com/mmuckley/torchkbnufft>`_.
# Below, we propose an implementation using the :class:`deepinv.physics.LinearPhysics`.
# As such, operations like grad and prox are available.

from deepinv.physics import LinearPhysics


class RadioInterferometry(LinearPhysics):
    r"""
    Radio Interferometry measurement operator.

    Args:
        img_size (tuple): Size of the target image, e.g., (H, W).
        samples_loc (torch.Tensor): Normalized sampling locations in the Fourier domain.
        dataWeight (torch.Tensor): Data weighting for the measurements.
        interp_points (Union[int, Sequence[int]]): Number of neighbors to use for interpolation in each dimension. Default is `7`.
        k_oversampling (float): Oversampling of the k space grid, should be between `1.25` and `2`. Default is `2`.
        real_projection (bool): Apply real projection after the adjoint NUFFT.
        device (torch.device): Device where the operator is computed.
    """

    def __init__(
        self,
        img_size,
        samples_loc,
        dataWeight=torch.tensor(
            [
                1.0,
            ]
        ),
        k_oversampling=2,
        interp_points=7,
        real_projection=True,
        device="cpu",
        **kwargs,
    ):
        super(RadioInterferometry, self).__init__(**kwargs)

        self.device = device
        self.k_oversampling = k_oversampling
        self.interp_points = interp_points
        self.img_size = img_size
        self.real_projection = real_projection

        # Check image size format
        assert len(self.img_size) == 2

        # Define oversampled grid
        self.grid_size = (
            int(img_size[0] * self.k_oversampling),
            int(img_size[1] * self.k_oversampling),
        )

        self.samples_loc = samples_loc.to(self.device)
        self.dataWeight = dataWeight.to(self.device)

        self.nufftObj = tkbn.KbNufft(
            im_size=self.img_size,
            grid_size=self.grid_size,
            numpoints=self.interp_points,
            device=self.device,
        )
        self.adjnufftObj = tkbn.KbNufftAdjoint(
            im_size=self.img_size,
            grid_size=self.grid_size,
            numpoints=self.interp_points,
            device=self.device,
        )

        # Define adjoint operator projection
        if self.real_projection:
            self.adj_projection = lambda x: torch.real(x).to(torch.float)
        else:
            self.adj_projection = lambda x: x

    def setWeight(self, w):
        self.dataWeight = w.to(self.device)

    def A(self, x):
        return (
            self.nufftObj(x.to(torch.cfloat), self.samples_loc, norm="ortho")
            * self.dataWeight
        )

    def A_adjoint(self, y):
        return self.adj_projection(
            self.adjnufftObj(y * self.dataWeight, self.samples_loc, norm="ortho")
        )


# %%
#
# This measurement operator is readily available in the Physics module in :class:`deepinv.physics.RadioInterferometry`
# and can be used directly as
#
# ::
#
#         from deepinv.physics import RadioInterferometry
#
#         physics = RadioInterferometry(img_size=img_size, samples_loc=samples_loc, device=device)
#
#
# Groundtruth image
# ----------------------------------------------------------------------------------------
# The following data is our groundtruth with the settings of Experiment II in :footcite:t:`aghabiglou2024r2d2`.
# The groundtruth data has been normalized in the [0, 1] range.
# As usual in radio interferometric imaging, the data has high dynamic range,
# i.e. the ratio between the faintest and highest emissions is higher than in traditional low-level vision tasks.
# In the case of this particular image, this ratio is of ``5000``.
# For this reason, unlike in other applications, we tend to visualize the logarithmic scale of the data instead of the data itself.

image_gdth = load_np_url(get_image_url("3c353_gdth.npy"))
image_gdth = torch.from_numpy(image_gdth).unsqueeze(0).unsqueeze(0).to(device)


def to_logimage(im, rescale=False, dr=5000):
    r"""
    A function plotting the image in logarithmic scale with specified dynamic range
    """
    if rescale:
        im = im - im.min()
        im = im / im.max()
    else:
        im = torch.clamp(im, 0, 1)
    return torch.log10(dr * im + 1.0) / np.log10(dr)


imgs = [image_gdth, to_logimage(image_gdth)]
plot(
    imgs,
    titles=[f"Groundtruth", f"Groundtruth \nin logarithmic scale"],
    cmap="inferno",
    cbar=True,
)

# %%
# Sampling pattern
# ----------------------------------------------------------------------------------------
# We'll load a simulated sampling pattern of `Very Large Array <https://public.nrao.edu/telescopes/vla/>`_ telescope.
# For simplicity, the coordinates of the sampling points have been normalized to the range of :math:`[-\pi, \pi]`.
# In RI imaging task, a super-resolution factor will normally be introduced in imaging step,
# so that the possibility of point sources appearing on the boundaries of pixels can be reduced.
# Here, this factor is ``1.5``.

uv = load_np_url(get_degradation_url("uv_coordinates.npy"))
uv = torch.from_numpy(uv).to(device)

scatter_plot([uv], titles=["uv coverage"], s=0.2, linewidths=0.0)

# %%
# Simulating the measurements
# ----------------------------------------------------------------------------------------
# We now have all the data and tools to generate our measurements!
# The noise level :math:`\tau` in the spacial Fourier domain is set to ``0.5976 * 2e-3``.
# This value will preserve the dynamic range of the groundtruth image in this case.
# Please check :footcite:t:`terris2023image` and :footcite:t:`aghabiglou2024r2d2`.
# for more information about the relationship between the noise level in the Fourier domain and the dynamic range of the target image.

tau = 0.5976 * 2e-3

# build sensing operator
physics = RadioInterferometry(
    img_size=image_gdth.shape[-2:],
    samples_loc=uv.permute((1, 0)),
    real_projection=True,
    device=device,
)

# Generate the physics
torch.manual_seed(0)
y = physics.A(image_gdth)
noise = (torch.randn_like(y) + 1j * torch.randn_like(y)) / np.sqrt(2)
y = y + tau * noise

# %%
# Natural weighting and Briggs weighting
# ----------------------------------------------------------------------------------------
# A common practice in RI consists is weighting the measurements in the Fourier domain to
# whiten the noise level in the spatial Fourier domain and compensate the over-sampling of visibilities at low-frequency regions.
# We here provide the Briggs-weighting scheme associated to the above uv-sampling pattern.

# load pre-computed Briggs weighting
nWimag = load_np_url(get_degradation_url("briggs_weight.npy"))
nWimag = torch.from_numpy(nWimag).reshape(1, 1, -1).to(device)

# apply natural weighting and Briggs weighting to measurements
y *= nWimag / tau

# add image weighting to the sensing operator
physics.setWeight(nWimag / tau)

# compute operator norm (note: increase the iteration number for higher precision)
opnorm = physics.compute_norm(
    torch.randn_like(image_gdth, device=device), max_iter=20, tol=1e-6, verbose=False
).item()
print("Operator norm: ", opnorm)

# %%
# The PSF, defined as :math:`\operatorname{PSF} = A \delta` (where :math:`\delta` is a Dirac), can be computed
# with the help of the :func:`deepinv.utils.dirac_like` function.

dirac = dirac_like(image_gdth).to(device)
PSF = physics.A_adjoint(physics.A(dirac))
print("PSF peak value: ", PSF.max().item())

psf_log = to_logimage(PSF, rescale=True)

plot_inset(
    [psf_log],
    titles=["PSF (logscale)"],
    cmap="viridis",
    extract_loc=(0.46, 0.46),
    extract_size=0.08,
    inset_loc=(0.0, 0.6),
    inset_size=0.4,
)

# %%
# The backprojected image :math:`A^{\top}Ay` is shown below.

back = physics.A_adjoint(y)

imgs = [to_logimage(image_gdth), to_logimage(back, rescale=True)]
plot(
    imgs,
    titles=[f"Groundtruth \n(logscale)", f"Backprojection \n(logscale)"],
    cmap="inferno",
)

# %%
# Solving the problem with a wavelet prior
# ----------------------------------------------------------------------------------------
# A traditional approach for solving the RI problem consists in solving the optimization problem
#
# .. math::
#   \begin{equation*}
#       \underset{x \geq 0}{\operatorname{min}} \,\, \frac{1}{2} \|Ax-y\|_2^2 + \lambda \sum_i \|\Psi_i x\|_{1}(x),
#   \end{equation*}
#
# where :math:`1/2 \|A(x)-y\|_2^2` is the a data-fidelity term, and each :math:`\|\Psi_i x\|_{1}(x)` is a sparsity
# inducing prior for the image :math:`x`, and :math:`\lambda>0` is a regularisation parameter. Simlarly to the `SARA <https://basp-group.github.io/BASPLib/SARA_family.html>`_
# algorithm, we use a dictionnary of 8 Daubechies wavelets as the prior.

from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import WaveletPrior

# Select the data fidelity term
data_fidelity = L2()

# Specify the prior (we redefine it with a smaller number of iteration for faster computation)
wv_list = ["db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8"]
prior = WaveletPrior(level=3, wv=wv_list, p=1, device="cpu", clamp_min=0)


# %%
# The problem is quite challenging and to reduce optimization time,
# we can start from an approximate guess of the solution that is pseudo-inverse reconstruction.


def custom_init(y, physics):
    x_init = torch.clamp(physics.A_dagger(y), 0)
    return {"est": (x_init, x_init)}


# %%
# We are now ready to implement the FISTA algorithm.
from deepinv.optim.optimizers import optim_builder

# Logging parameters
verbose = True

plot_convergence_metrics = (
    True  # compute performance and convergence metrics along the algorithm.
)

# Algorithm parameters
stepsize = 1.0 / (1.5 * opnorm)
lamb = 1e-3 * opnorm  # wavelet regularisation parameter
params_algo = {"stepsize": stepsize, "lambda": lamb}
max_iter = 50
early_stop = True

# Instantiate the algorithm class to solve the problem.
model = optim_builder(
    iteration="FISTA",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=verbose,
    params_algo=params_algo,
    custom_init=custom_init,
)

# reconstruction with FISTA algorithm
x_model, metrics = model(y, physics, x_gt=image_gdth, compute_metrics=True)

# compute PSNR
print(
    f"Linear reconstruction PSNR: {dinv.metric.PSNR()(image_gdth, back).item():.2f} dB"
)
print(
    f"FISTA reconstruction PSNR: {dinv.metric.PSNR()(image_gdth, x_model).item():.2f} dB"
)

# plot images
# sphinx_gallery_multi_image = "single"
imgs = [
    to_logimage(image_gdth),
    to_logimage(back, rescale=True),
    to_logimage(x_model, rescale=True),
]
# %%
plot(imgs, titles=["GT", "Linear", "Recons."], cmap="inferno", cbar=True)

# plot convergence curves
if plot_convergence_metrics:
    plot_curves(metrics)

# %%
# We can see that the bright sources are generally recovered,
# but not for the faint and extended emissions.
# We kindly point the readers to `BASPLib <https://basp-group.github.io/BASPLib/>`_
# for the state-of-the-art RI imaging algorithms, such as
# `R2D2 <https://basp-group.github.io/BASPLib/R2D2.html>`_,
# `AIRI <https://basp-group.github.io/BASPLib/AIRI.html>`_,
# `SARA <https://basp-group.github.io/BASPLib/SARA_family.html>`_,
# and corresponding reconstructions.

# %%
# :References:
#
# .. footbibliography::
