.. _models:

Models
======
This package provides vanilla signal reconstruction networks, which can be used for a quick evaluation of a learning setting.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.Denoiser
   deepinv.models.ArtifactRemoval
   deepinv.models.DeepImagePrior

Classical Denoisers
-------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.WaveletPrior
   deepinv.models.WaveletDict
   deepinv.models.TGV
   deepinv.models.MedianFilter


Learnable Denoisers
-------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.AutoEncoder
   deepinv.models.UNet
   deepinv.models.ConvDecoder

The following denoisers have **pretrained weights** available.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.DnCNN
   deepinv.models.DRUNet
   deepinv.models.GSDRUNet