Models
==============================
This package provides vanilla signal reconstruction networks, which can be used for a quick evaluation of a learning setting.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.Denoiser
   deepinv.models.ArtifactRemoval

Classical Denoisers
--------------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.WaveletPrior
   deepinv.models.WaveletDict
   deepinv.models.TGV


Learned Denoisers
--------------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.AutoEncoder
   deepinv.models.UNet
   deepinv.models.DnCNN
   deepinv.models.DRUNet
   deepinv.models.GSDRUNet


..
    this is a code snippet showing how to load a denoiser