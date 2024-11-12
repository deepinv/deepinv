deepinv.models
===============

Classical Denoisers
-------------------

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


Deep Denoisers
-------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.AutoEncoder
   deepinv.models.UNet
   deepinv.models.DnCNN
   deepinv.models.DRUNet
   deepinv.models.SCUNet
   deepinv.models.GSDRUNet
   deepinv.models.SwinIR
   deepinv.models.DiffUNet
   deepinv.models.Restormer
   deepinv.models.ICNN



Denoisers Utils
--------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.EquivariantDenoiser
   deepinv.models.complex.to_complex_denoiser
   deepinv.models.TimeAgnosticNet
   deepinv.models.TimeAveragingNet



Artifact Removal
--------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.ArtifactRemoval

Deep Image Prior
--------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.DeepImagePrior
   deepinv.models.ConvDecoder


Adversarial Networks
--------------------
Discriminator networks used in networks trained with adversarial learning using :ref:`adversarial losses <adversarial-losses>`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.PatchGANDiscriminator
   deepinv.models.ESRGANDiscriminator
   deepinv.models.DCGANGenerator
   deepinv.models.DCGANDiscriminator
   deepinv.models.CSGMGenerator
