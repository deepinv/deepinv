.. _pretrained-weights:

Pretrained Weights
------------------

The following denoisers have **pretrained weights** available; we next briefly summarize the origin of the weights,
associated reference and relevant details. All pretrained weights are hosted on
`HuggingFace <https://huggingface.co/deepinv>`_.


.. list-table:: Summary of pretrained weights
   :widths: 25 25
   :header-rows: 1

   * - Model
     - Weight
   * - :class:`deepinv.models.DnCNN`
     - from `Learning Maximally Monotone Operators <https://github.com/matthieutrs/LMMO_lightning>`_
       trained on noise level 2.0/255. `DnCNN grayscale weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_gray.pth?download=true>`_, `DnCNN color weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_color.pth?download=true>`_.
   * -
     - from `Learning Maximally Monotone Operators <https://github.com/matthieutrs/LMMO_lightning>`_ with Lipschitz
       constraint to ensure approximate firm nonexpansiveness, trained on noise level 2.0/255. `Non-expansive DnCNN grayscale weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_lipschitz_gray.pth?download=true>`_, `Non-expansive DnCNN color weights <https://huggingface.co/deepinv/dncnn/resolve/main/dncnn_sigma2_lipschitz_color.pth?download=true>`_.
   * - :class:`deepinv.models.DRUNet`
     - Default: trained with deepinv `(logs) <https://wandb.ai/matthieu-terris/drunet?workspace=user-matthieu-terris>`_, trained on noise levels in [0, 20]/255
       and on the same dataset as DPIR `DRUNet grayscale weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_gray.pth?download=true>`_, `DRUNet color weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_deepinv_color.pth?download=true>`_.
   * -
     - from `DPIR <https://github.com/cszn/DPIR>`_,
       trained on noise levels in [0, 50]/255. `DRUNet original grayscale weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_gray.pth?download=true>`_, `DRUNET original color weights <https://huggingface.co/deepinv/drunet/resolve/main/drunet_color.pth?download=true>`_.
   * - :class:`deepinv.models.GSDRUNet`
     - weights from `Gradient-Step PnP <https://github.com/samuro95/GSPnP>`_, trained on noise levels in [0, 50]/255.
       `GSDRUNet color weights <https://huggingface.co/deepinv/gradientstep/blob/main/GSDRUNet.ckpt>` and `GSDRUNet grayscale weights <https://huggingface.co/deepinv/gradientstep/blob/main/GSDRUNet_grayscale_torch.ckpt>`.
   * - :class:`deepinv.models.SCUNet`
     - from `SCUNet <https://github.com/cszn/SCUNet>`_,
       trained on images degraded with synthetic realistic noise and camera artefacts. `SCUNet color weights <https://huggingface.co/deepinv/scunet/resolve/main/scunet_color_real_psnr.pth?download=true>`_.
   * - :class:`deepinv.models.SwinIR`
     - from `SwinIR <https://github.com/JingyunLiang/SwinIR>`_, trained on various noise levels levels in {15, 25, 50}/255, in color and grayscale.
       The weights are automatically downloaded from the authors' `project page <https://github.com/JingyunLiang/SwinIR/releases>`_.
   * - :class:`deepinv.models.DiffUNet`
     - Default: from `Ho et al. <https://arxiv.org/abs/2108.02938>`_ trained on FFHQ (128 hidden channels per layer).
       `DiffUNet weights <https://huggingface.co/deepinv/diffunet/resolve/main/diffusion_ffhq_10m.pt?download=true>`_.
   * -
     - from `Dhariwal and Nichol <https://arxiv.org/abs/2105.05233>`_ trained on ImageNet128 (256 hidden channels per layer).
       `weights <https://huggingface.co/deepinv/diffunet/resolve/main/diffusion_openai.pt?download=true>`_.
   * - :class:`deepinv.models.EPLLDenoiser`
     - Default: parameters estimated with deepinv on 50 mio patches from the training/validation images from BSDS500 for grayscale and color images.
   * -
     - Code for generating the weights for the example :ref:`patch-prior-demo` is contained within the demo
   * - :class:`deepinv.models.Restormer`
     - from `Restormer: Efficient Transformer for High-Resolution Image Restoration <https://arxiv.org/abs/2111.09881>`_. Pretrained parameters from `swz30 github <https://github.com/swz30/Restormer/tree/main>`_.
   * -
     - Also available on the `deepinverse Restormer HugginfaceHub <https://huggingface.co/deepinv/Restormer/tree/main>`_.