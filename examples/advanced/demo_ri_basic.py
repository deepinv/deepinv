"""
A simple example for RI imaging task
====================================================================================================

In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. 
The following example and data are taken from `Aghabiglou et al. (2024) <https://arxiv.org/abs/2403.05452>`_. 
If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check `BASPLib <https://basp-group.github.io/BASPLib/>`_.

"""

# %%
# Setup paths/url for data loading.
# ----------------------------------------------------------------------------------------
PTH_GDTH = '/Users/chao/Documents/GitHub/deepinv_ri_example/examples/advanced/data/3c353_gdth.npy'
PTH_UV = '/Users/chao/Documents/GitHub/deepinv_ri_example/examples/advanced/data/uv_coordinates.npy'
PTH_WEIGHT = '/Users/chao/Documents/GitHub/deepinv_ri_example/examples/advanced/data/briggs_weight.npy'

# %%
# Import required packages
# ----------------------------------------------------------------------------------------
# We rely on the `TorchKbNufft` as the non-uniform FFT backend in this problem.
# This first snippet is just here to check that dependencies are installed properly.

import torch
import numpy as np
import torchkbnufft as tkbn

import deepinv as dinv
from deepinv.utils.plotting import plot, plot_curves
from deepinv.utils.demo import load_np_url

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
# Below, we propose an implementation wrapping it in :class:`deepinv.physics.LinearPhysics`. 
# As such, operations like grad and prox are available.

from deepinv.physics import LinearPhysics

class MeasOpRI(LinearPhysics):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def __init__(
        self,
        img_size,
        samples_loc,
        dataWeight=torch.tensor([1.0,]),
        device=torch.device('cpu'),
        **kwargs
    ):
        super(MeasOpRI, self).__init__(**kwargs)
        
        self.device = device
        self.nufftObj = tkbn.KbNufft(im_size=img_size, numpoints=7, device=device)
        self.adjnufftObj = tkbn.KbNufftAdjoint(im_size=img_size, numpoints=7, device=device)
        self.samples_loc = samples_loc.to(device)
        self.dataWeight = dataWeight.to(device=device)

    def setWeight(self, w):
        self.dataWeight = w.to(device=device)

    def A(self, x):
        return self.nufftObj(x.to(torch.complex64), self.samples_loc) * self.dataWeight

    def A_adjoint(self, y):
        return torch.real(self.adjnufftObj(y*self.dataWeight, self.samples_loc)).to(torch.float)

# %%
# Groundtruth image
# ----------------------------------------------------------------------------------------
# The following data is our groundtruth with the settings of Experiment II in `Aghabiglou et al. (2024) <https://arxiv.org/abs/2403.05452>`_.. 
# The groundtruth data has been normalized in the [0, 1] range. 
# As usual in radio interferometric imaging, the data has high dynamic range, 
# i.e. the ratio between the faintest and highest emissions is higher than in traditional low-level vision task. 
# In the case of this particular image, this ratio is of ``5000``. 
# For this reason, unlike in other applications, we tend to visualize the logarithmic scale of the data instead of the data itself.

image_gdth = np.load(PTH_GDTH) #load_np_url(PTH_GDTH)
image_gdth = torch.from_numpy(image_gdth).unsqueeze(0).unsqueeze(0)

def to_logimage(im, rescale=False, dr=5000):
    r"""
    A function plotting the image in logarithmic scale with specified dynamic range
    """
    if rescale:
        im = im-im.min()
        im = im/im.max()
    else:
        im = torch.clamp(im, 0, 1)
    return torch.log10(dr*im + 1.)/np.log10(dr)

imgs = [image_gdth, to_logimage(image_gdth)]
plot(
    imgs,
    titles=[f"Groundtruth", f"Groundtruth in logarithmic scale"],
    cmap='inferno'
)

# %%
# Sampling pattern
# ----------------------------------------------------------------------------------------
# We'll load a simulated sampling pattern of `Very Large Array <https://public.nrao.edu/telescopes/vla/>`_ telescope. 
# For simplicity, the coordinates of the sampling points have been normalized to the range of :math:`[-\pi, \pi]`. 
# In RI imaging task, a super-resolution factor will normally be introduced in imaging step, 
# so that the possibility of point sources appearing on the boundaries of pixels can be reduced. 
# Here, this factor is ``1.5``.

import matplotlib.pyplot as plt  # TODO: add scatter-plot utils in deepinv?

uv = np.load(PTH_UV) #load_np_url(PTH_UV)
uv = torch.from_numpy(uv)

plt.figure(figsize=(3,3))
plt.scatter(uv[:,0], uv[:,1], s=0.001)
plt.ylabel('v')
plt.xlabel('u')
plt.axis('square')
plt.xlim((-np.pi,np.pi))
plt.ylim((-np.pi,np.pi))
plt.title('Sampling trajectory in the Fourier domain')
plt.show()

# %%
# Simulating the measurements
# ----------------------------------------------------------------------------------------
# We now have all the data and tools to generate our measurements! 
# The noise level :math:`\tau` in the spacial Fourier domain is set to ``0.5976``. 
# This value will preserve the dynamic range of the groundtruth image in this case. 
# Please check `Terris et al. (2024) <https://doi.org/10.1093/mnras/stac2672>`_ and `Aghabiglou et al. (2024) <https://arxiv.org/abs/2403.05452>`_
# for more information about the relationship between the noise level in the Fourier domain and the dynamic range of the target image.

tau = 0.5976

# build sensing operator
physics = MeasOpRI(
    img_size=image_gdth.shape[-2:],
    samples_loc=uv.permute((1,0)),
    real=True,
    device=device)

# Generate the physics
y = physics.A(image_gdth)
noise = (torch.randn_like(y)+1j*torch.randn_like(y))/np.sqrt(2)
y = y + tau*noise

# %%
# Natural weighting and Briggs weighting
# ----------------------------------------------------------------------------------------
# A common practice in RI consists is weighting the measurements in the Fourier domain to 
# whiten the noise level in the spatial Fourier domain and compensate the over-sampling of visibilities at low-frequency regions. 
# We here provide the Briggs-weighting scheme associated to the above uv-sampling pattern.

# load pre-computed Briggs weighting
nWimag = np.load(PTH_WEIGHT) #load_np_url(PTH_WEIGHT)
nWimag = torch.from_numpy(nWimag).reshape(1,1,-1).to(device)

# apply natural weighting and Briggs weighting to measurements
y *= nWimag/tau

# add image weighting to the sensing operator
physics.setWeight(nWimag/tau)

# compute operator norm (note: increase the iteration number for higher precision)
opnorm = physics.compute_norm(
    torch.randn_like(image_gdth, device=device),
    max_iter=20,
    tol=1e-6,
    verbose=False).item()
print('Operator norn: ', opnorm)

##############################
# The PSF, defined as :math:`\operatorname{PSF} = A \delta` (where :math:`\delta` is a Dirac), can be computed as follows.

# compute PSF  # TODO: add Dirac util
dirac = torch.zeros_like(image_gdth)
dirac[0,0,image_gdth.shape[-2]//2,image_gdth.shape[-1]//2] = 1.0
PSF = physics.A_adjoint(physics.A(dirac))
print('PSF peak value: ', PSF.max().item())

plot(
    to_logimage(PSF, rescale=True),
    titles=f"PSF (log scale)",
    cmap='viridis'
)

##############################
# The backprojected image :math:`A^{\top}Ay` is shown below.

back = physics.A_adjoint(y)

imgs = [to_logimage(image_gdth), to_logimage(back, rescale=True)]
plot(
    imgs,
    titles=[f"Groundtruth (logscale)", f"Backprojection (logscale)"],
    cmap='inferno'
)

# %%
# Solving the problem with a wavelet prior
# ----------------------------------------------------------------------------------------
# A traditional approach for solving the RI problem consists in solving the optimization problem
# 
# .. math::
#   \begin{equation*}
#       \underset{x \geq 0}{\operatorname{min}} \,\, \frac{1}{2} \|Ax-y\|_2^2 + \lambda \|\Psi x\|_{1}(x),
#   \end{equation*}
# 
# where :math:`1/2 \|A(x)-y\|_2^2` is the a data-fidelity term, :math:`\|\Psi x\|_{1}(x)` is a sparsity inducing
# prior for the image :math:`x`, and :math:`\lambda>0` is a regularisation parameter.
#
# TODO: 
# * the block below will be removed to add a small paramter in the wavelet prior with clamp to 0
# * add SARA convex prior

from deepinv.optim import Prior
from deepinv.models.wavdict import WaveletDenoiser

class PositiveWaveletPrior(Prior):
    r"""
    Wavelet prior :math:`\reg{x} = \|\Psi x\|_{p}`.

    :math:`\Psi` is an orthonormal wavelet transform, and :math:`\|\cdot\|_{p}` is the :math:`p`-norm, with
    :math:`p=0`, :math:`p=1`, or :math:`p=\infty`.

    .. note::
        Following common practice in signal processing, only detail coefficients are regularized, and the approximation
        coefficients are left untouched.

    .. warning::
        For 3D data, the computational complexity of the wavelet transform cubically with the size of the support. For
        large 3D data, it is recommended to use wavelets with small support (e.g. db1 to db4).


    :param int level: level of the wavelet transform. Default is 3.
    :param str wv: wavelet name to choose among those available in `pywt <https://pywavelets.readthedocs.io/en/latest/>`_. Default is "db8".
    :param float p: :math:`p`-norm of the prior. Default is 1.
    :param str device: device on which the wavelet transform is computed. Default is "cpu".
    :param int wvdim: dimension of the wavelet transform, can be either 2 or 3. Default is 2.
    """

    def __init__(self, level=3, wv="db8", p=1, device="cpu", wvdim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.p = p
        self.wv = wv
        self.wvdim = wvdim
        self.level = level
        self.device = device
        if p == 0:
            self.non_linearity = "hard"
        elif p == 1:
            self.non_linearity = "soft"
        elif p == np.inf or p == "inf":
            self.non_linearity = "topk"
        else:
            raise ValueError("p should be 0, 1 or inf")
        self.WaveletDenoiser = WaveletDenoiser(
            level=self.level,
            wv=self.wv,
            device=self.device,
            non_linearity=self.non_linearity,
            wvdim=self.wvdim,
        )

    def g(self, x, *args, reduce=True, **kwargs):
        r"""
        Computes the regularizer

        .. math::
            \begin{equation*}
                {\regname}_{i,j}(x) = \|(\Psi x)_{i,j}\|_{p}
            \end{equation*}


        where :math:`\Psi` is an orthonormal wavelet transform, :math:`i` and :math:`j` are the indices of the
        wavelet sub-bands,  and :math:`\|\cdot\|_{p}` is the :math:`p`-norm, with
        :math:`p=0`, :math:`p=1`, or :math:`p=\infty`. As mentioned in the class description, only detail coefficients
        are regularized, and the approximation coefficients are left untouched.

        If `reduce` is set to `True`, the regularizer is summed over all detail coefficients, yielding

        .. math::
            \begin{equation*}
                \regname(x) = \|\Psi x\|_{p}.
            \end{equation*}

        If `reduce` is set to `False`, the regularizer is returned as a list of the norms of the detail coefficients.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :param bool reduce: if True, the prior is summed over all detail coefficients. Default is True.
        :return: (torch.Tensor) prior :math:`g(x)`.
        """
        list_dec = self.psi(x)
        list_norm = torch.hstack([torch.norm(dec, p=self.p) for dec in list_dec])
        if reduce:
            return torch.sum(list_norm)
        else:
            return list_norm

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""Compute the proximity operator of the wavelet prior with the denoiser :class:`~deepinv.models.WaveletDenoiser`.
        Only detail coefficients are thresholded.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """
        out = self.WaveletDenoiser(x, ths=gamma)
        return torch.clamp(out, 0, 1)

    def psi(self, x):
        r"""
        Applies the (flattening) wavelet decomposition of x.
        """
        return self.WaveletDenoiser.psi(x, self.wv, self.level, self.wvdim)

##############################
# The problem is quite challenging and to reduce optimization time, 
# we can start from an approximate guess of the solution that is the backprojected image divided by the PSF peak.

def custom_init(y, physics):
    x_init = torch.clamp(physics.A_dagger(y), 0)
    return {"est": (x_init, x_init)}

init_variables = custom_init(y, physics)
init_image = init_variables['est'][0]

# plot images. Images are saved in RESULTS_DIR.
imgs = [to_logimage(init_image)]
plot(
    imgs,
    titles=["Initialization"],
    cmap='inferno'
)

##############################
# We are now ready to implement the FISTA algorithm.

from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder

# Select the data fidelity term
data_fidelity = L2()

# Specify the prior (we redefine it with a smaller number of iteration for faster computation)
# prior = dinv.optim.prior.WaveletPrior(level=3, wv='db8', p=1, device='cpu')
prior = PositiveWaveletPrior(level=3, wv='db8', p=1, device='cpu')

# Logging parameters
verbose = True
plot_metrics = True  # compute performance and convergence metrics along the algorithm, curved saved in RESULTS_DIR

# Algorithm parameters
stepsize = 1.0/(1.5*opnorm)
lamb = 1.0  # wavelet regularisation parameter
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
    custom_init=custom_init
)

# reconstruction with FISTA algorithm
with torch.no_grad():
    x_model, metrics = model(
        y, physics, x_gt=image_gdth, compute_metrics=True
    )  

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.utils.metric.cal_psnr(image_gdth, back):.2f} dB")
print(f"FISTA reconstruction PSNR: {dinv.utils.metric.cal_psnr(image_gdth, x_model):.2f} dB")

# plot images. Images are saved in RESULTS_DIR.
imgs = [to_logimage(image_gdth), to_logimage(back, rescale=True), to_logimage(x_model, rescale=True)]
plot(
    imgs,
    titles=["GT", "Linear", "Recons."],
    cmap='inferno'
)

# plot convergence curves
if plot_metrics:
    plot_curves(metrics)

##############################
# We can see that the bright sources are generally recovered, 
# but not for the faint and extended emissions. 
# We kindly point the readers to `BASPLib <https://basp-group.github.io/BASPLib/>`_ 
# for the state-of-the-art RI imaging algorithms, such as 
# `R2D2 <https://basp-group.github.io/BASPLib/R2D2.html>`_, 
# `AIRI <https://basp-group.github.io/BASPLib/AIRI.html>`_, 
# `SARA <https://basp-group.github.io/BASPLib/SARA_family.html>`_,
# and corresponding reconstructions.