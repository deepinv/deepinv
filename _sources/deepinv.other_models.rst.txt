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


Networks for time-varying data
------------------------------
When using time-varying (i.e. dynamic) data of 5D shape (B,C,T,H,W), the reconstruction network must be adapted.
To adapt any existing network to take dynamic data as independent time-slices, create a time-agnostic wrapper that 
flattens the time dimension into the batch dimension.


.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.models.TimeAgnosticNet
   deepinv.models.TimeAveragingNet