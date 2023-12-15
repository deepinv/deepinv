.. _models:

Models
======
This package provides vanilla signal reconstruction methods,
which can be used for a quick evaluation of a learning setting.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.ArtifactRemoval
   deepinv.models.DeepImagePrior

Denoisers
---------
Denoisers are :class:`torch.nn.Module` that take a noisy image as input and return a denoised image.
They can be used as a building block for plug-and-play restoration, for building unrolled architectures,
or as a standalone denoiser. All denoisers have a ``forward`` method that takes a noisy image and a noise level
(which generally corresponds to the standard deviation of the noise) as input and returns a denoised image.

.. note::

    Some denoisers (e.g., :class:`deepinv.models.DnCNN`) do not use the information about the noise level.
    In this case, the noise level is ignored.

Classical Denoisers
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.BM3D
   deepinv.models.MedianFilter
   deepinv.models.TV
   deepinv.models.TGV
   deepinv.models.WaveletPrior
   deepinv.models.WaveletDict


Learnable Denoisers
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.AutoEncoder
   deepinv.models.ConvDecoder
   deepinv.models.UNet
   deepinv.models.DnCNN
   deepinv.models.DRUNet
   deepinv.models.SCUNet
   deepinv.models.GSDRUNet
   deepinv.models.SwinIR
   deepinv.models.DiffUNet


Equivariant denoisers
---------------------
The denoisers can be turned into equivariant denoisers by wrapping them with the
:class:`deepinv.models.EquivariantDenoiser` class.
The group of transformations available at the moment are vertical/horizontal flips, 90 degree rotations, or a
combination of both, consisting in groups with 3, 4 or 8 elements.

The denoising can either be averaged the group of transformation (making the denoiser equivariant) or performed on a
single transformation sampled uniformly at random in the group, making the denoiser a Monte-Carlo estimator of the exact
equivariant denoiser.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.EquivariantDenoiser


Unfolded architectures
----------------------
Some more specific unfolded architectures are also available.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.PDNet_PrimalBlock
   deepinv.models.PDNet_DualBlock


.. _pretrained-weights:
Pretrained weights
------------------
The following denoisers have **pretrained weights** available; we next briefly summarize the origin of the weights,
associated reference and relevant details. All pretrained weights are hosted on
`HuggingFace <https://huggingface.co/deepinv>`_.


.. list-table:: Summary of pretrained weights
   :widths: 25 25
   :header-rows: 1

   * - Model
     - Weight
   * - :meth:`deepinv.models.DnCNN`
     - from `Learning Maximally Monotone Operators <https://github.com/matthieutrs/LMMO_lightning>`_
       trained on noise level 2.0/255. `[grayscale weights] <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_gray.pth?download=true>`_ `[color weights] <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_color.pth?download=true>`_.
   * -
     - from `Learning Maximally Monotone Operators <https://github.com/matthieutrs/LMMO_lightning>`_ with Lipschitz
       constraint to ensure approximate firm nonexpansiveness, trained on noise level 2.0/255. `[grayscale weights] <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_lipschitz_gray.pth?download=true>`_ `[color weights] <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_lipschitz_color.pth?download=true>`_.
   * - :meth:`deepinv.models.DRUNet`
     - Default: trained with deepinv `(logs) <https://wandb.ai/matthieu-terris/drunet?workspace=user-matthieu-terris>`_, trained on noise levels in [0, 20]/255
       and on the same dataset as DPIR `[grayscale weights] <https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_gray.pth?download=true>`_, `[color weights] <https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_color.pth?download=true>`_.
   * -
     - from `DPIR <https://github.com/cszn/DPIR>`_,
       trained on noise levels in [0, 50]/255. `[grayscale weights] <https://huggingface.co/deepinv/drunet/resolve/main/drunet_gray.pth?download=true>`_, `[color weights] <https://huggingface.co/deepinv/drunet/resolve/main/drunet_color.pth?download=true>`_.
   * - :meth:`deepinv.models.GSDRUNet`
     - weights from `Gradient-Step PnP <https://github.com/samuro95/GSPnP>`_, trained on noise levels in [0, 50]/255.
       `[color weights] <https://huggingface.co/deepinv/gradientstep/blob/main/GSDRUNet.ckpt>`_.
   * - :meth:`deepinv.models.SCUNet`
     - from `SCUNet <https://github.com/cszn/SCUNet>`_,
       trained on images degraded with synthetic realistic noise and camera artefacts. `[color weights] <https://huggingface.co/deepinv/scunet/resolve/main/scunet_color_real_psnr.pth?download=true>`_.
   * - :meth:`deepinv.models.SwinIR`
     - from `SwinIR <https://github.com/JingyunLiang/SwinIR>`_, trained on various noise levels levels in {15, 25, 50}/255, in color and grayscale.
       The weights are automatically downloaded from the authors' `project page <https://github.com/JingyunLiang/SwinIR/releases>`_.
   * - :meth:`deepinv.models.DiffUNet`
     - Default: from `Ho et al. <https://arxiv.org/abs/2108.02938>`_ trained on FFHQ (128 hidden channels per layer).
       `[weights] <https://huggingface.co/deepinv/diffunet/resolve/main/diffusion_ffhq_10m.pt?download=true>`_.
   * -
     - from `Dhariwal and Nichol <https://arxiv.org/abs/2105.05233>`_ trained on ImageNet128 (256 hidden channels per layer).
       `[weights] <https://huggingface.co/deepinv/diffunet/resolve/main/diffusion_openai.pt?download=true>`_.

