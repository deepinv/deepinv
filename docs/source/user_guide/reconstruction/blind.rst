.. _blind:

Blind Inverse Problems
======================

Following the :ref:`notation of the library <parameter-dependent-operators>`, here we consider measurements of the form
:math:`y = \noise{\forw{x, \theta}}`, where :math:`\theta` represents unknown physics parameters.
Noise parameters associated to :math:`\noise{\cdot}` may also be unknown. In this section, we consider two classes of problems:

- **Calibration problems**: Estimate the unknown parameters :math:`\theta` given paired signal and measurement data :math:`(x,y)`

- **Blind inverse problems**: Jointly estimate the signal :math:`x` and :math:`\theta` parameters (and other noise parameters) from the measurements :math:`y`
Some methods directly estimate the signal without explicitly estimating the parameters.

Calibration problems
~~~~~~~~~~~~~~~~~~~~
If paired measurement and signal data is available at inference time, physics parameters can be estimated using optimization methods.
See the example :class:`sphx_glr_auto_examples_blind-inverse-problems_demo_optimizing_physics_parameters.py` for more details.

Estimating physics parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If only measurement data is available :math:`\theta` at inference time, we can estimate the parameters from the observed data,
and then use any non-blind reconstructor to recover the image.
The library provides the following parameter estimation models/algorithms:

.. list-table:: Identification models
   :widths: 15 15 15 20 20
   :header-rows: 1

   * - Model/Algorithm
     - Tensor Size (C, H, W)
     - Pretrained Weights
     - Physics
     - Parameters estimated
     - Examples

   * - :class:`deepinv.models.KernelIdentificationNetwork`
     - C=3; H,W>8
     - RGB
     - :class:`deepinv.physics.SpaceVaryingBlur`
     - `filters`, `multipliers`
     - :ref:`blind deblurring <sphx_glr_auto_examples_blind-inverse-problems_demo_blind_deblurring.py>`.

   * - :func:`ESPIRiT <deepinv.MultiCoilMRI.estimate_coil_maps>`
     - C=2; H,W>64
     - (non-learned)
     - :class:`deepinv.physics.MultiCoilMRI`
     - `coil_maps`
     - :ref:`blind deblurring <sphx_glr_auto_examples_physics_demo_mri_tour.py>`.