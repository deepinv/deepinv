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


Pretrained weights
------------------
The following denoisers have **pretrained weights** available; we next briefly summarize the origin of the weights,
associated reference and relevant details.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.DnCNN

- weights from `Learning Maximally Monotone Operators <https://github.com/matthieutrs/LMMO_lightning>`_,
  trained on noise level 2.0/255. `[grayscale weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=dncnn_sigma2_gray.pth>`_ `[color weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=dncnn_sigma2_color.pth>`_.
- weights from `Learning Maximally Monotone Operators <https://github.com/matthieutrs/LMMO_lightning>`_ with Lipschitz
  constraint to ensure approximate firm nonexpansiveness, trained on noise level 2.0/255. `[grayscale weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=dncnn_sigma2_lipschitz_gray.pth>`_ `[color weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=dncnn_sigma2_lipschitz_color.pth>`_.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.DRUNet

- weights from `DPIR <https://github.com/cszn/DPIR>`_,
  trained on noise levels in [0, 20]/255. `[grayscale weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=drunet_gray.pth>`_ `[color weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=drunet_color.pth>`_.
- weights trained with deepinv `(logs) <https://wandb.ai/matthieu-terris/drunet?workspace=user-matthieu-terris>`_, trained on noise levels in [0, 20]/255
  and on the same dataset as DPIR. `[color weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=dncnn_sigma2_lipschitz_color.pth>`_.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.SCUNet

- weights from `SCUNet <https://github.com/cszn/SCUNet>`_,
  trained on images degraded with synthetic realistic noise and camera artefacts. `[color weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=scunet_color_real_psnr.pth>`_.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.GSDRUNet

- weights from `Gradient-Step PnP <https://github.com/samuro95/GSPnP>`_, trained on noise levels in [0, 20]/255.
  `[color weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=GSDRUNet.ckpt>`_.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.SwinIR


- weights from `SwinIR <https://github.com/JingyunLiang/SwinIR>`_, trained on various noise levels levels in {15, 25, 50}/255, in color and grayscale.
  The weights are automatically downloaded from the authors' `project page <https://github.com/JingyunLiang/SwinIR/releases>`_.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.DiffUNet

- weights from `Ho et al. <https://arxiv.org/abs/2108.02938>`_ trained on FFHQ (128 hidden channels per layer).
  `[weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=diffusion_ffhq_10m.pt>`_.
- weights from `Dhariwal and Nichol <https://arxiv.org/abs/2105.05233>`_ trained on ImageNet128 (256 hidden channels per layer).
  `[weights] <https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fweights&files=diffusion_openai.pt>`_.



