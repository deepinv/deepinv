.. _denoisers:

Denoisers
==================

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

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   deepinv.denoisers.classical
   deepinv.denoisers.learned
   deepinv.denoisers.utils


.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :link: classical-denoisers
        :link-type: ref

        :octicon:`zap` **Classical denoisers**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Classical denoisers which do not require training, such as BM3D or TV.

    .. grid-item-card::
        :link: deep-denoisers
        :link-type: ref

        :octicon:`flame` **Deep denoisers**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Deep architectures pretrained for denoising, and
        to be used as building blocks for reconstruction nets.


    .. grid-item-card::
        :link: denoiser-utils
        :link-type: ref

        :octicon:`briefcase` **Utils**
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Utilities for denoisers, such as complex denoisers or equivariant denoisers.



