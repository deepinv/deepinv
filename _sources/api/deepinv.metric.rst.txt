deepinv.metric
===============

Metrics are generally used to evaluate the performance of a model, or as the distance function inside a loss function.
Please refer to the :ref:`user guide <metric>` for more information.

Base class
----------
.. userguide:: metric

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.metric.Metric


Full Reference Metrics
----------------------
.. userguide:: full-reference-metrics

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.metric.MSE
    deepinv.loss.metric.NMSE
    deepinv.loss.metric.MAE
    deepinv.loss.metric.PSNR
    deepinv.loss.metric.SSIM
    deepinv.loss.metric.L1L2
    deepinv.loss.metric.LpNorm
    deepinv.loss.metric.LPIPS
    deepinv.loss.metric.SpectralAngleMapper
    deepinv.loss.metric.ERGAS
    deepinv.loss.metric.HaarPSI


No Reference Metrics
--------------------
.. userguide:: no-reference-metrics

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.loss.metric.NIQE
    deepinv.loss.metric.QNR
