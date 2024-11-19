.. _artifact:

Artifact Removal
================
The simplest method for reconstructing an image from a measurements is to first map the measurements
to the image domain via a non-learned mapping, and then apply a denoiser network to the obtain the final reconstruction.
This idea was introduced by Jin et al. `"Deep Convolutional Neural Network for Inverse Problems in Imaging" <https://ieeexplore.ieee.org/abstract/document/7949028>`_
for tomographic reconstruction.

The :class:`deepinv.models.ArtifactRemoval` class implements networks as

.. math::

    \inversef{x,A} = \phi(A^{\dagger}y)

where :math:`A^{\dagger}` is a non-learned pseudo-inverse of the forward operator (e.g., adjoint operator
if :math:`A` is linear, the identity if :math:`y` already lies in the image domain, etc.)
:math:`\phi` is the denoiser/backbone network (see a :ref:`list of available architectures <deep-architectures>`),
and :math:`y` are the measurements.