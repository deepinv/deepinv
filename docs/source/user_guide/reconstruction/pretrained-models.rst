.. _pretrained-models:

Pretrained Models
=================

Pretrained reconstructors
~~~~~~~~~~~~~~~~~~~~~~~~~

Some methods do not require any training and can be quickly deployed to your problem.

.. seealso::

  See the example :ref:`sphx_glr_auto_examples_basics_demo_pretrained_model.py` for how to get started with these models on various problems.

These models can be set-up in one line and perform inference in another line:

.. doctest::

  >>> import deepinv as dinv
  >>> x = dinv.utils.load_example("butterfly.png")
  >>> physics = dinv.physics.Downsampling(filter="bicubic")
  >>> y = physics(x)
  >>> model = dinv.models.RAM(pretrained=True) # or any of the models listed below
  >>> x_hat = model(y, physics) # Model inference
  >>> dinv.metric.PSNR()(x_hat, x)
  tensor([31.9825])

.. list-table:: Pretrained reconstructors
   :header-rows: 1

   * - **Name**
     - **Family**
     - **Modality/Physics**
     - **Speed**
   * - :class:`RAM <deepinv.models.RAM>`
     - Feedforward
     - General; physics must be linear
     - Fast
   * - :class:`DPIR <deepinv.optim.DPIR>`
     - :ref:`Plug-and-play <iterative>` w/ pretrained denoiser
     - General
     - Medium
   * - :class:`DDRM <deepinv.sampling.DDRM>`
     - :ref:`Diffusion <diffusion>` w/ pretrained denoiser
     - General; physics must be decomposable
     - Slow
   * - :class:`DiffPIR <deepinv.sampling.DiffPIR>`
     - :ref:`Diffusion <diffusion>` w/ pretrained denoiser
     - General
     - Slow
   * - :class:`DPS <deepinv.sampling.DPS>`
     - :ref:`Diffusion <diffusion>` w/ pretrained denoiser
     - General
     - Slow
   * - :ref:`Pretrained denoisers <pretrained-weights>`
     - Feedforward
     - Denoising
     - Fast

.. tip::

  If you want to fine-tune these models on your own measurements (without or with ground truth) or physics, check out :ref:`sphx_glr_auto_examples_models_demo_foundation_model.py`.

.. seealso::

  See below for a full list of denoisers that can be plugged into iterative/sampling algorithms.

.. _pretrained-weights:

Description of weights
~~~~~~~~~~~~~~~~~~~~~~

For each model (:class:`Denoiser <deepinv.models.Denoiser>` or :class:`Reconstructor <deepinv.models.Reconstructor>`) that has pretrained weights, we briefly summarize the origin of the weights,
associated reference and relevant details. All pretrained weights are hosted on
`HuggingFace <https://huggingface.co/deepinv>`_.

Click on the model name to learn more about the type of model and use `pretrained=True` to use the pretrained weights.

.. list-table:: Summary of pretrained weights
   :widths: 20 5 25
   :header-rows: 1

   * - Model
     - Type
     - Weight
   * - :class:`deepinv.models.DnCNN`
     - Denoiser
     - Default weights from `Learning Maximally Monotone Operators <https://github.com/matthieutrs/LMMO_lightning>`_
       trained on noise level 2.0/255:
       `DnCNN grayscale weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_gray.pth?download=true>`_, `DnCNN color weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_color.pth?download=true>`_.
   * -
     -
     - Alternative weights trained on noise level 2.0/255 with Lipschitz constraint to ensure approximate firm nonexpansiveness:
       `Non-expansive DnCNN grayscale weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_lipschitz_gray.pth?download=true>`_, `Non-expansive DnCNN color weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_lipschitz_color.pth?download=true>`_.
   * - :class:`deepinv.models.DRUNet`
     - Denoiser
     - Default weights trained with deepinv `(logs) <https://wandb.ai/matthieu-terris/drunet?workspace=user-matthieu-terris>`_, trained on noise levels in [0, 20]/255
       and on the same dataset as DPIR:
       `DRUNet grayscale weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_gray.pth?download=true>`_, `DRUNet color weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_color.pth?download=true>`_.
   * -
     -
     - Alternative weights from `DPIR <https://github.com/cszn/DPIR>`_,
       trained on noise levels in [0, 50]/255. `DRUNet original grayscale weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_gray.pth?download=true>`_, `DRUNET original color weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_color.pth?download=true>`_.
   * - :class:`deepinv.models.GSDRUNet`
     - Denoiser
     - Weights from `Gradient-Step PnP <https://github.com/samuro95/GSPnP>`_, trained on noise levels in [0, 50]/255:
       `GSDRUNet color weights <https://huggingface.co/deepinv/gradientstep/blob/main/GSDRUNet.ckpt>`_ and `GSDRUNet grayscale weights <https://huggingface.co/deepinv/gradientstep/blob/main/GSDRUNet_grayscale_torch.ckpt>`_.
   * - :class:`deepinv.models.SCUNet`
     - Denoiser
     - Weights from `SCUNet <https://github.com/cszn/SCUNet>`_,
       trained on images degraded with synthetic realistic noise and camera artefacts. `SCUNet color weights <https://huggingface.co/deepinv/scunet/resolve/main/scunet_color_real_psnr.pth?download=true>`_.
   * - :class:`deepinv.models.SwinIR`
     - Denoiser
     - Weights from `SwinIR <https://github.com/JingyunLiang/SwinIR>`_, trained on various noise levels levels in {15, 25, 50}/255, in color and grayscale.
       The weights are automatically downloaded from the authors' `project page <https://github.com/JingyunLiang/SwinIR/releases>`_.
   * - :class:`deepinv.models.DiffUNet`
     - Denoiser
     - Default weights from `Ho et al. <https://arxiv.org/abs/2108.02938>`_ trained on FFHQ (128 hidden channels per layer):
       `DiffUNet weights <https://huggingface.co/deepinv/diffunet/resolve/main/diffusion_ffhq_10m.pt?download=true>`_.
   * -
     -
     - Alternative weights from `Dhariwal and Nichol <https://arxiv.org/abs/2105.05233>`_ trained on ImageNet128 (256 hidden channels per layer):
       `DiffUNet OpenAI weights <https://huggingface.co/deepinv/diffunet/resolve/main/diffusion_openai.pt?download=true>`_.
   * - :class:`deepinv.models.EPLLDenoiser`
     - Denoiser
     - Weights estimated with deepinv on 50 mio patches from the training/validation images from BSDS500 for grayscale and color images.
       Code for generating the weights for the example :ref:`patch-prior-demo` is contained within the demo.
   * - :class:`deepinv.models.Restormer`
     - Denoiser
     - Weights from `Restormer: Efficient Transformer for High-Resolution Image Restoration <https://arxiv.org/abs/2111.09881>`_:
       `Restormer weights <https://github.com/swz30/Restormer/tree/main>`_,
       also available on the `deepinverse Restormer HuggingfaceHub <https://huggingface.co/deepinv/Restormer/tree/main>`_.
   * - :class:`deepinv.models.RAM`
     - Reconstructor & Denoiser
     - Weights from `Terris et al. <https://github.com/matthieutrs/ram>`_ :footcite:p:`terris2025reconstruct`. Pretrained weights from `RAM HuggingfaceHub <https://huggingface.co/mterris/ram>`_.
