deepinv.transform
=================

This module contains different transforms which can be used for data augmentation or together with the equivariant losses.
Please refer to the :ref:`user guide <transform>` for more information.

Base class
----------
.. userguide:: transform

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.transform.Transform

Simple transforms
-----------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.transform.Rotate
    deepinv.transform.Shift
    deepinv.transform.Scale
    deepinv.transform.Reflect

Advanced transforms
-------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.transform.Homography
    deepinv.transform.projective.Euclidean
    deepinv.transform.projective.Similarity
    deepinv.transform.projective.Affine
    deepinv.transform.projective.PanTiltRotate
    deepinv.transform.CPABDiffeomorphism

Video transforms
----------------

While all geometric transforms accept video input, the following transforms work specifically in the time dimension.
These can be easily compounded with geometric transformations using the ``*`` operation.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.transform.ShiftTime