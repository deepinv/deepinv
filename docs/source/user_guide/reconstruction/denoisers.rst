.. _denoisers:

Denoisers
=========

The :class:`deepinv.models.Denoiser` base class describe
denoisers that take a noisy image as input and return a denoised image.
They can be used as a building block for plug-and-play restoration, for building unrolled architectures,
for artifact removal networks, or as a standalone denoisers. All denoisers have a
:func:`forward <deepinv.models.Denoiser.forward>` method that takes a
noisy image and a noise level (which generally corresponds to the standard deviation of the noise)
as input and returns a denoised image:

    >>> import torch
    >>> import deepinv as dinv
    >>> denoiser = dinv.models.DRUNet()
    >>> sigma = 0.1
    >>> image = torch.ones(1, 3, 32, 32) * .5
    >>> noisy_image =  image + torch.randn(1, 3, 32, 32) * sigma
    >>> denoised_image = denoiser(noisy_image, sigma)

.. note::

    Some denoisers (e.g., :class:`deepinv.models.DnCNN`) do not use the information about the noise level.
    In this case, the noise level is ignored.

.. _deep-architectures:

Deep denoisers
~~~~~~~~~~~~~~
We provide the following list of deep denoising architectures,
which are based on CNN, Transformer or hybrid CNN-Transformer modules.
See :ref:`pretrained-weights` for more information on pretrained denoisers.

.. list-table:: Deep denoisers
   :widths: 15 25 15 15 10
   :header-rows: 1

   * - Model
     - Type
     - Tensor Size (C, H, W)
     - Pretrained Weights
     - Noise level aware
   * - :class:`deepinv.models.AutoEncoder`
     - Fully connected
     - Any
     - No
     - No
   * - :class:`deepinv.models.UNet`
     - CNN
     - Any C; H,W>8
     - No
     - No
   * - :class:`deepinv.models.DnCNN`
     - CNN
     - Any C, H, W
     - RGB, grayscale
     - No
   * - :class:`deepinv.models.DRUNet`
     - CNN-UNet
     - Any C; H,W>8
     - RGB, grayscale
     - Yes
   * - :class:`deepinv.models.GSDRUNet`
     - CNN-UNet
     - Any C; H,W>8
     - RGB, grayscale
     - Yes
   * - :class:`deepinv.models.SCUNet`
     - CNN-Transformer
     - Any C, H, W
     - No
     - No
   * - :class:`deepinv.models.SwinIR`
     - CNN-Transformer
     - Any C, H, W
     - RGB
     - No
   * - :class:`deepinv.models.DiffUNet`
     - Transformer
     - Any C; H,W = 64, 128, 256, ...
     - RGB
     - Yes
   * - :class:`deepinv.models.Restormer`
     - CNN-Transformer
     - Any C, H, W
     - RGB, grayscale, deraining, deblurring
     - No
   * - :class:`deepinv.models.ICNN`
     - CNN
     - Any C; H, W = 128, 256,...
     - No
     - No


.. _non-learned-denoisers:

Classical denoisers
~~~~~~~~~~~~~~~~~~~
All denoisers in this list are non-learned (except for EPLL)
and rely on hand-crafted priors.

.. list-table:: Non-Learned Denoisers Overview
   :widths: 30 30 30
   :header-rows: 1

   * - Model
     - Info
     - Tensor Size (C, H, W)
   * - :class:`deepinv.models.BM3D`
     - Patch-based denoiser
     - C=1 or C=3, any H, W.
   * - :class:`deepinv.models.MedianFilter`
     - Non-learned filter
     - Any C, H, W
   * - :class:`deepinv.models.TVDenoiser`
     - :class:`Total variation prior <deepinv.optim.TVPrior>`
     - Any C, H, W
   * - :class:`deepinv.models.TGVDenoiser`
     - Total generalized variation prior
     - Any C, H, W
   * - :class:`deepinv.models.WaveletDenoiser`
     - :class:`Sparsity in orthogonal wavelet domain <deepinv.optim.WaveletPrior>`
     - Any C, H, W
   * - :class:`deepinv.models.WaveletDictDenoiser`
     - Sparsity in overcomplete wavelet domain
     - Any C, H, W
   * - :class:`deepinv.models.EPLLDenoiser`
     - Learned patch-prior
     - C=1 or C=3, any H, W

.. _denoiser-utils:

Denoisers Utilities
~~~~~~~~~~~~~~~~~~~

Equivariant denoisers
^^^^^^^^^^^^^^^^^^^^^
Denoisers can be turned into equivariant denoisers by wrapping them with the
:class:`deepinv.models.EquivariantDenoiser` class, which symmetrizes the denoiser
with respect to a transform from our :ref:`available transforms <transform>` such as :class:`deepinv.transform.Rotate`
or :class:`deepinv.transform.Reflect`. You retain full flexibility by passing in the transform of choice.
The denoising can either be averaged over the entire group of transformation (making the denoiser equivariant) or
performed on 1 or n transformations sampled uniformly at random in the group, making the denoiser a Monte-Carlo
estimator of the exact equivariant denoiser.

Complex denoisers
^^^^^^^^^^^^^^^^^
Most denoisers in the library are designed to process real images. However, some problems, e.g., phase retrieval,
require processing complex-valued images. The function :class:`deepinv.models.complex.to_complex_denoiser` can convert any real-valued denoiser into
a complex-valued denoiser. It can be simply called by ``complex_denoiser = to_complex_denoiser(denoiser)``.

Dynamic networks
^^^^^^^^^^^^^^^^
When using time-varying (i.e. dynamic) data of 5D shape (B,C,T,H,W), the reconstruction network must be adapted
using :class:`deepinv.models.TimeAveragingNet`.

To adapt any existing network to take dynamic data as independent time-slices, :class:`deepinv.models.TimeAgnosticNet`
creates a time-agnostic wrapper that flattens the time dimension into the batch dimension.

