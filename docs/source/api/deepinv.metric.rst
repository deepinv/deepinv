deepinv.metric
===============

This package contains popular metrics for inverse problems.
Metrics are generally used to evaluate the performance of a model, or as the distance function inside a loss function.

Base class
-----------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.metric.Metric


Distortion metrics
------------------

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
        deepinv.loss.metric.LpNorm


Perceptual metrics
------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

        deepinv.loss.metric.LPIPS
        deepinv.loss.metric.NIQE
