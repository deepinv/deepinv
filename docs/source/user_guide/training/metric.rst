.. _metric:

Metrics
=======

This package contains popular metrics for inverse problems.

Metrics are generally used to evaluate the performance of a model, or as the distance function inside a loss function.

Introduction
------------
All metrics inherit from the base class :class:`deepinv.loss.metric.Metric`, which is a :class:`torch.nn.Module`.
All metrics take either ``x_net, x`` for a full-reference metric or ``x_net`` for a no-reference metric.

All metrics can perform a standard set of pre and post processing, including
operating on complex numbers, normalisation and reduction. See :class:`deepinv.loss.metric.Metric` for more details.

.. note::

    By default, metrics do not reduce over the batch dimension, as the usual usage is to average the metrics over a dataset yourself.
    This discourages averaging over metrics which might in turn have averaged over uneven batch sizes.
    Note we provide :class:`deepinv.utils.AverageMeter` to easily keep track of the average of metrics.
    For example, we use this in our trainer :class:`deepinv.Trainer`.

    However, you can use the ``reduction`` argument to perform reduction, e.g. if you want a single metric calculation rather than over a dataset.

All metrics can either be used directly as metrics, or as the backbone for training losses.
To do this, wrap the metric in a suitable loss such as :class:`deepinv.loss.SupLoss` or :class:`deepinv.loss.MCLoss`.
In this way, :class:`deepinv.loss.metric.MSE` replaces :class:`torch.nn.MSELoss` and :class:`deepinv.loss.metric.MAE` replaces :class:`torch.nn.L1Loss`,
and you can use these in a loss like ``SupLoss(metric=MSE())``.

Metrics can be classified as distortion or perceptual,
see `the Perception-Distortion Tradeoff <https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf>`_
for an explanation of distortion vs perceptual metrics.

Finally, you can also wrap existing metric functions using ``Metric(metric=f)``, see :class:`deepinv.loss.metric.Metric` for an example.

.. note::

    For some metrics, higher is better; for these, you must also set ``train_loss=True``.

.. tip::

    For convenience, you can also import metrics directly from ``deepinv.metric`` or ``deepinv.loss``.

Example:

.. doctest::

    >>> import torch
    >>> import deepinv as dinv
    >>> m = dinv.metric.SSIM()
    >>> x = torch.ones(2, 3, 16, 16) # B,C,H,W
    >>> x_hat = x + 0.01
    >>> m(x_hat, x) # Calculate metric for each image in batch
    tensor([1.0000, 1.0000])
    >>> m = dinv.metric.SSIM(reduction="sum")
    >>> m(x_hat, x) # Sum over batch
    tensor(1.9999)
    >>> l = dinv.loss.MCLoss(metric=dinv.metric.SSIM(train_loss=True, reduction="mean")) # Use SSIM for training

.. _full-reference-metrics:

Full Reference Metrics
----------------------
Full reference metrics are used to measure the difference between the original ``x`` and the reconstructed image ``x_net``.

.. list-table:: Full Reference Metrics
   :header-rows: 1

   * - **Metric**
     - **Definition**

   * - :class:`deepinv.loss.metric.MSE`
     - :math:`\text{MSE}(\hat{x},x) = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{x}_i)^2`

   * - :class:`deepinv.loss.metric.NMSE`
     - :math:`\text{NMSE}(\hat{x},x) = \frac{\| x - \hat{x} \|_2^2}{\| x \|_2^2}`

   * - :class:`deepinv.loss.metric.MAE`
     - :math:`\text{MAE}(\hat{x},x) = \frac{1}{n} \sum_{i=1}^n |x_i - \hat{x}_i|`

   * - :class:`deepinv.loss.metric.PSNR`
     - :math:`\text{PSNR}(\hat{x},x) = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}(\hat{x},x)} \right)`, where :math:`\text{MAX}` is the maximum possible pixel value of the image

   * - :class:`deepinv.loss.metric.SSIM`
     - :math:`\text{SSIM}(\hat{x},x) = \frac{(2 \mu_x \mu_{\hat{x}} + C_1)(2 \sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}`, where :math:`\mu` and :math:`\sigma` are mean and variance

   * - :class:`deepinv.loss.metric.L1L2`
     - :math:`\text{L1L2}(\hat{x},x) = \alpha \|x - \hat{x}\|_1 + (1 - \alpha) \|x - \hat{x}\|_2`, where :math:`\alpha` is a balancing parameter

   * - :class:`deepinv.loss.metric.LpNorm`
     - :math:`\text{LpNorm}(\hat{x},x) = \|x - \hat{x}\|_p^p`

   * - :class:`deepinv.loss.metric.LPIPS`
     - Uses a pretrained network to calculate the perceptual similarity between two images.

   * - :class:`deepinv.loss.metric.SpectralAngleMapper`
     - Multispectral image metric that calculates spectral similarity between bands.

   * - :class:`deepinv.loss.metric.ERGAS`
     - "Error relative global dimensionless synthesis" multispectral image metric for pan-sharpening problems.



.. _no-reference-metrics:

No Reference Metrics
--------------------

We implement no-reference perceptual metrics, they only require the reconstructed image ``x_net``.

.. list-table:: No Reference Metrics
   :header-rows: 1

   * - **Metric**
     - **Definition**

   * - :class:`deepinv.loss.metric.NIQE`
     - Calculates deviation of image from statistical regularities of natural images.

   * - :class:`deepinv.loss.metric.QNR`
     - Multispectral image metric :math:`\text{QNR}(\hat{x}) = (1-D_\lambda)^\alpha(1 - D_s)^\beta`, where :math:`D_\lambda` and :math:`D_s` are spectral and spatial distortions.
