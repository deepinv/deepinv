deepinv.models
===============

This module contains a collection of models for denoising and reconstruction.
Please refer to the :ref:`user guide <user_guide>` for more information.

Base Classes
------------
.. userguide:: reconstructors

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.Denoiser
   deepinv.models.Reconstructor


Classical Denoisers
-------------------
.. userguide:: non-learned-denoisers

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
--------------
.. userguide:: deep-architectures

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
   deepinv.models.VarNet
   deepinv.models.PanNet



Denoisers Utils
---------------
.. userguide:: denoiser-utils

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.EquivariantDenoiser
   deepinv.models.TimeAgnosticNet
   deepinv.models.TimeAveragingNet

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.models.complex.to_complex_denoiser

Artifact Removal
----------------
.. userguide:: artifact

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.ArtifactRemoval

Deep Image Prior
----------------
.. userguide:: deep-image-prior

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.DeepImagePrior
   deepinv.models.ConvDecoder


Adversarial Networks
--------------------
.. userguide:: adversarial-losses

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.PatchGANDiscriminator
   deepinv.models.ESRGANDiscriminator
   deepinv.models.DCGANGenerator
   deepinv.models.DCGANDiscriminator
   deepinv.models.CSGMGenerator
