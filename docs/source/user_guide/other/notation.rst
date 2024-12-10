Math Notation
=============

The documentation of ``deepinv`` uses a unified mathematical notation that is summarized in the following table:

.. list-table:: List of mathematical symbols
   :widths: 10 50
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`x\in\xset`
     - Underlying image or signal to reconstruct of :math:`n` elements.
   * - :math:`y\in\yset`
     - Observed measurement vector of size :math:`m`.
   * - :math:`p(x)`
     - Distribution of images :math:`x` (often referred to as prior distribution).
   * - :math:`p(y)`
     - Distribution of measurements :math:`y`.
   * - :math:`\forw{x}`
     - Deterministic mapping that captures the physics of the imaging system.
   * - :math:`A^\top\colon\yset\to\xset`
     - Adjoint of the measurement operator.
   * - :math:`\noise{y}`
     - Stochastic mapping adding noise to measurements.
   * - :math:`\inverse{y}`
     - Reconstruction network that maps measurements to images :math:`y\mapsto x`.
   * - :math:`\denoiser{x}{\sigma}`
     - Gaussian denoiser for noise of standard deviation :math:`\sigma`.
   * - :math:`\datafid{x}{y} = \distance{A(x)}{y}`
     - Data fidelity term, enforcing measurement consistency :math:`y\approx A(x)`, depending on the choice of the
       distance function (see below).
   * - :math:`\distance{u}{y}`
     - Distance function measuring the discrepancy between :math:`u` and :math:`y`.
       It is linked to the noise model (likelihood).
   * - :math:`\reg{x}`
     - Regularization term that promotes plausible reconstructions. It is linked to :math:`p(x)`.
   * - :math:`\lambda`
     - Hyperparameter controlling the amount of regularization.
