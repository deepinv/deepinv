.. _classical-denoisers:

Classical Denoisers
-------------------

We provide a set of classical denoisers that use hand-crafted priors to denoise images, and do not require any training.
The following denoisers are available:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.BM3D
   deepinv.models.MedianFilter
   deepinv.models.TVDenoiser
   deepinv.models.TGVDenoiser
   deepinv.models.WaveletDenoiser
   deepinv.models.WaveletDictDenoiser
   deepinv.models.EPLLDenoiser