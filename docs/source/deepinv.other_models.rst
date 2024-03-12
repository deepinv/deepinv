Other Reconstruction Methods
==============================
This module provides alternative methods for reconstructing images from measurements, which are not well described
by other modules in the library.


Learned Filtered Back-Projection
--------------------------------------
The simplest method for reconstructing an image from a measurements is to first map the measurements
to the image domain via a non-learned mapping, and then apply a deep network to the obtain the final reconstruction.
This idea was introduced by Jin et al. `"Deep Convolutional Neural Network for Inverse Problems in Imaging" <https://ieeexplore.ieee.org/abstract/document/7949028>`_
for tomographic reconstruction.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.ArtifactRemoval

