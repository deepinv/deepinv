.. _denoiser-utils:

Equivariant Denoisers
--------------------------
The denoisers can be turned into equivariant denoisers by wrapping them with the
:class:`deepinv.models.EquivariantDenoiser` class, which symmetrizes the denoiser
with respect to a transform from our :ref:`available transforms <transform>` such as :class:`deepinv.transform.Rotate` or :class:`deepinv.transform.Reflect`.
You retain full flexibility by passing in the transform of choice.

The denoising can either be averaged over the entire group of transformation (making the denoiser equivariant) or performed on 1 or n
transformations sampled uniformly at random in the group, making the denoiser a Monte-Carlo estimator of the exact
equivariant denoiser.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.EquivariantDenoiser


Complex Denoisers
--------------------------

Most denoisers in the library are designed to process real images. However, some problems, e.g., phase retrieval, require processing complex-valued images.The function :class:`deepinv.models.complex.to_complex_denoiser` can convert any real-valued denoiser into a complex-valued denoiser. It can be simply called by ``complex_denoiser = to_complex_denoiser(denoiser)``.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.complex.to_complex_denoiser

