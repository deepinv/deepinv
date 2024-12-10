.. _artifact:

Artifact Removal
================
The simplest method for reconstructing an image from a measurements is to first map the measurements
to the image domain via a non-learned mapping, and then apply a denoiser network to the obtain the final reconstruction.


The :class:`deepinv.models.ArtifactRemoval` class converts a denoiser :class:`deepinv.models.Denoiser` or other image-to-image network :math:`\phi` into a
reconstruction network :class:`deepinv.models.Reconstructor` :math:`R` by doing

- | Adjoint: :math:`\inversef{y}{A}=\phi(A^{\top}y)` with ``mode='adjoint'``.
  | This option is generally to linear operators :math:`A`.
- Pseudoinverse: :math:`\inversef{y}{A}=\phi(A^{\dagger}y)` with ``mode='pinv'``.
- | Direct: :math:`\inversef{y}{A}=\phi(y)` with ``mode='direct'``.
  | This option serves as a wrapper to obtain a :class:`Reconstructor <deepinv.models.Reconstructor>`, and can be used to adapt a generic denoiser or image-to-image network into one that is specific to an inverse problem.
