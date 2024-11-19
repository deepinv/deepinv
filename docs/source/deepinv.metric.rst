.. _metric:

Metrics
=======

This package contains popular metrics for inverse problems. 

Metrics are generally used to evaluate the performance of a model, or as the distance function inside a loss function.

Introduction
--------------------
All metrics inherit from the base class :meth:`deepinv.loss.metric.Metric`, which is a :meth:`torch.nn.Module`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.metric.Metric


All metrics take either ``x_net, x``
for a full-reference metric or ``x_net`` for a no-reference metric.

All metrics can perform a standard set of pre and post processing, including
operating on complex numbers, normalisation and reduction. See :class:`deepinv.loss.metric.Metric` for more details.

.. note::

    By default, metrics do not reduce over the batch dimension, as the usual usage is to average the metrics over a dataset yourself.
    However, you can use the ``reduction`` argument to perform reduction, e.g. if the metric is to be used as a training loss.

All metrics can either be used directly as metrics, or as the backbone for training losses.
To do this, wrap the metric in a suitable loss such as :class:`deepinv.loss.SupLoss` or :class:`deepinv.loss.MCLoss`.
In this way, :class:`deepinv.loss.metric.MSE` replaces :class:`torch.nn.MSELoss` and :class:`deepinv.loss.metric.MAE` replaces :class:`torch.nn.L1Loss`,
and you can use these in a loss like ``SupLoss(metric=MSE())``.

.. note::

    For some metrics, higher is better; for these, you must also set ``train_loss=True``.

.. note::

    For convenience, you can also import metrics directly from ``deepinv.metric`` or ``deepinv.loss``.

Finally, you can also wrap existing metric functions using ``Metric(metric=f)``, see :class:`deepinv.loss.metric.Metric` for an example.

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

Distortion metrics
------------------

We implement popular distortion metrics
(see `The Perception-Distortion Tradeoff <https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf>`_
for an explanation of distortion vs perceptual metrics):

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.loss.metric.MSE
        deepinv.loss.metric.NMSE
        deepinv.loss.metric.MAE
        deepinv.loss.metric.PSNR
        deepinv.loss.metric.SSIM
        deepinv.loss.metric.QNR
        deepinv.loss.metric.L1L2

Perceptual metrics
------------------

We implement popular perceptual metrics:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:
    
        deepinv.loss.metric.LPIPS
        deepinv.loss.metric.NIQE

Utils
-------
A set of popular distances that can be used by the supervised and self-supervised losses.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.metric.LpNorm