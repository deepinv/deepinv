.. _denoisers:

Denoisers
=========

Denoisers are :class:`torch.nn.Module` that take a noisy image as input and return a denoised image.
They can be used as a building block for plug-and-play restoration, for building unrolled architectures,
for artifact removal networks, or as a standalone denoisers. All denoisers have a ``forward`` method that takes a noisy image and a noise level
(which generally corresponds to the standard deviation of the noise) as input and returns a denoised image:

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

.. list-table:: Deep denoisers
   :widths: 15 25 15 15
   :header-rows: 1

   * - Model
     - Type
     - Image Size
     - Pretrained Weights
   * - :class:`deepinv.models.AutoEncoder`
     - CNN
     - 256x256
     - Yes
   * - :class:`deepinv.models.UNet`
     - CNN
     - 256x256
     - Yes
   * - :class:`deepinv.models.DnCNN`
     - CNN
     - 256x256
     - Yes
   * - :class:`deepinv.models.DRUNet`
     - CNN
     - 256x256
     - Yes
   * - :class:`deepinv.models.SCUNet`
     - CNN
     - 512x512
     - No
   * - :class:`deepinv.models.GSDRUNet`
     - CNN
     - 256x256
     - Yes
   * - :class:`deepinv.models.SwinIR`
     - Transformer
     - 512x512
     - Yes
   * - :class:`deepinv.models.DiffUNet`
     - CNN
     - 256x256
     - No
   * - :class:`deepinv.models.Restormer`
     - Transformer
     - 512x512
     - Yes
   * - :class:`deepinv.models.ICNN`
     - CNN
     - 128x128
     - No

.. list-table:: Non-Learned Denoisers Overview
   :widths: 20 20 20 15
   :header-rows: 1

   * - Model
     - Info (Type: Non-learned Filter or Algorithm)
     - Image Size
     - Channels
   * - :class:`deepinv.models.BM3D`
     - Non-learned filter
     - Variable
     - Grayscale
   * - :class:`deepinv.models.MedianFilter`
     - Non-learned filter
     - Variable
     - Grayscale/RGB
   * - :class:`deepinv.models.TVDenoiser`
     - Non-learned, Total Variation
     - Variable
     - Grayscale/RGB
   * - :class:`deepinv.models.TGVDenoiser`
     - Non-learned, Total Generalized Variation
     - Variable
     - Grayscale/RGB
   * - :class:`deepinv.models.WaveletDenoiser`
     - Non-learned, Wavelet-based
     - Variable
     - Grayscale/RGB
   * - :class:`deepinv.models.WaveletDictDenoiser`
     - Non-learned, Wavelet Dictionary-based
     - Variable
     - Grayscale/RGB
   * - :class:`deepinv.models.EPLLDenoiser`
     - Non-learned, EPLL (Expected Patch Log Likelihood)
     - Variable
     - Grayscale/RGB


Denoisers Utilities
-------------------
The denoisers can be turned into equivariant denoisers by wrapping them with the
:class:`deepinv.models.EquivariantDenoiser` class, which symmetrizes the denoiser
with respect to a transform from our :ref:`available transforms <transform>` such as :class:`deepinv.transform.Rotate`
or :class:`deepinv.transform.Reflect`. You retain full flexibility by passing in the transform of choice.

The denoising can either be averaged over the entire group of transformation (making the denoiser equivariant) or
performed on 1 or n transformations sampled uniformly at random in the group, making the denoiser a Monte-Carlo
estimator of the exact equivariant denoiser.

Most denoisers in the library are designed to process real images. However, some problems, e.g., phase retrieval,
require processing complex-valued images.
The function :class:`deepinv.models.complex.to_complex_denoiser` can convert any real-valued denoiser into
a complex-valued denoiser. It can be simply called by ``complex_denoiser = to_complex_denoiser(denoiser)``.


Networks for time-varying data
------------------------------
When using time-varying (i.e. dynamic) data of 5D shape (B,C,T,H,W), the reconstruction network must be adapted.
To adapt any existing network to take dynamic data as independent time-slices, create a time-agnostic wrapper that
flattens the time dimension into the batch dimension.
