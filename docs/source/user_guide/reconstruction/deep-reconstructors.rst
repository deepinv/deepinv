.. _deep-reconstructors:

Deep Reconstruction Models
==========================

The simplest method for reconstructing an image from measurements is to pass it through a feedforward
model architecture that is conditioned on the acquisition physics, that is :math:`\inversef{y}{A}`. We offer a range of architectures for general and specific problems.

.. _artifact:

Artifact Removal
~~~~~~~~~~~~~~~~

The simplest reconstruction architecture first maps the measurements
to the image domain via a non-learned mapping, and then applys a denoiser network to the obtain the final reconstruction.

The :class:`deepinv.models.ArtifactRemoval` class converts a denoiser :class:`deepinv.models.Denoiser` or other image-to-image network :math:`\phi` into a
reconstruction network :class:`deepinv.models.Reconstructor` :math:`R` by doing

- | Adjoint: :math:`\inversef{y}{A}=\phi(A^{\top}y)` with ``mode='adjoint'``.
  | This option is generally to linear operators :math:`A`.
- Pseudoinverse: :math:`\inversef{y}{A}=\phi(A^{\dagger}y)` with ``mode='pinv'``.
- | Direct: :math:`\inversef{y}{A}=\phi(y)` with ``mode='direct'``.
  | This option serves as a wrapper to obtain a :class:`Reconstructor <deepinv.models.Reconstructor>`, and can be used to adapt a generic denoiser or image-to-image network into one that is specific to an inverse problem.


.. _general-reconstructors:

General reconstruction models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We provide the following list of reconstruction models trained on multiple various physics and datasets
to provide robustness to different problems.

See :ref:`pretrained-weights` for more information on pretrained denoisers.

.. list-table:: Multiphysics reconstruction models
   :widths: 15 25 15 15 10
   :header-rows: 1

   * - Model
     - Type
     - Tensor Size (C, H, W)
     - Pretrained Weights
     - Noise level aware
   * - :class:`deepinv.models.RAM`
     - CNN-UNet
     - C=1, 2, 3; H,W>8
     - C=1, 2, 3
     - Yes

.. _specific-reconstructors:

Specific reconstruction models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide some architectures for specific inverse problems.

.. list-table:: Specific architectures
   :header-rows: 1

   * - Model
     - Description
   * - :class:`deepinv.models.PanNet`
     - PanNet model for pansharpening.