.. _blind:

Blind Inverse Problems
======================

Blind inverse problems involve the simultaneous estimation of both the target signal and unknown parameters of the forward model, such as blur kernels or noise parameters.
Following the :ref:`notation of the library <parameter-dependent-operators>`, we consider measurements of the form :math:`y = \noise{\forw{x, \theta}}`, where :math:`\theta` represents unknown physics parameters.
Noise parameters associated to :math:`\noise{\cdot}` may also be unknown.
The goal is to jointly estimate the signal :math:`x` and the forward operator :math:`\theta` parameters (and other noise parameters) from the measurements :math:`y`
Some methods directly estimate the signal without explicitly estimating the parameters.

Estimating physics parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In many imaging problems, the physics parameters (e.g. blur kernel) are unknown and must be estimated alongside the image.
The library provides models that can estimate the parameters from the observed data, and then use any reconstructor
(e.g., :class:`RAM <deepinv.models.RAM>`, :class:`DPIR <deepinv.optim.DPIR>`, etc.) to recover the image.

.. seealso::

  See the example :ref:`sphx_glr_auto_examples_blind-inverse-problems_demo_blind_deblurring.py`.

.. list-table:: Identification models
   :widths: 15 15 15 20 20
   :header-rows: 1

   * - Model
     - Tensor Size (C, H, W)
     - Pretrained Weights
     - Physics
     - Parameters estimated

   * - :class:`deepinv.models.KernelIdentificationNetwork`
     - C=3; H,W>8
     - RGB
     - :class:`deepinv.physics.SpaceVaryingBlur`
     - `filters`, `multipliers`