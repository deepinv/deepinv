.. _transform:

Transforms
============

This package contains different transforms which can be used for data augmentation or together with the equivariant losses.

We implement various geometric transforms, ranging from Euclidean to homography and diffeomorphisms, some of which offer group-theoretic properties.

**See** :ref:`sphx_glr_auto_examples_basics_demo_transforms.py` **for example usage and visualisations.**

Transforms inherit from :class:`deepinv.transform.Transform`. Transforms can also be stacked by summing them, chained by multiplying them (i.e. product group), or joined via ``|`` to randomly select.
There are numerous other parameters e.g to randomly transform multiple times at once, to constrain the parameters to a range etc.

Transforms can also be used to make a denoiser equivariant using :class:`deepinv.models.EquivariantDenoiser` by performing Reynolds averaging using ``symmetrize()``. 
They can also be used for equivariant imaging (EI) using the :class:`deepinv.loss.EILoss` loss.
See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_ei_transforms.py` and :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_equivariant_imaging.py` for examples.

If needed, transforms can also be made deterministic by passing in specified parameters to the forward method.
This allows every transform to have its own deterministic inverse using ``transform.inverse()``.
Transforms can also be seamlessly integrated with existing ``torchvision`` transforms.
Transforms can also accept video (5D) input.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.transform.Transform

For example, random transforms can be used as follows:

.. doctest::

    >>> import torch
    >>> from deepinv.transform import Shift, Rotate
    >>> x = torch.rand((1, 1, 2, 2)) # Define random image (B,C,H,W)
    >>> transform = Shift() # Define random shift transform
    >>> transform(x).shape
    torch.Size([1, 1, 2, 2])
    >>> y = transform(transform(x, x_shift=[1]), x_shift=[-1]) # Deterministic transform
    >>> torch.all(x == y)
    tensor(True)
    >>> transform(torch.rand((1, 1, 3, 2, 2))).shape # Accepts video input of shape (B,C,T,H,W)
    torch.Size([1, 1, 3, 2, 2])
    >>> transform = Rotate() + Shift() # Stack rotate and shift transforms
    >>> transform(x).shape
    torch.Size([2, 1, 2, 2])
    >>> rotoshift = Rotate() * Shift() # Chain rotate and shift transforms
    >>> rotoshift(x).shape
    torch.Size([1, 1, 2, 2])
    >>> transform = Rotate() | Shift() # Randomly select rotate or shift transforms
    >>> transform(x).shape
    torch.Size([1, 1, 2, 2])
    >>> f = lambda x: x[..., [0]] * x # Function to be symmetrized
    >>> f_s = rotoshift.symmetrize(f)
    >>> f_s(x).shape
    torch.Size([1, 1, 2, 2])


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

We implement the following further geometric transforms.
The projective transformations formulate the image transformations using the pinhole camera model, from which various transformation subgroups can be derived. 
See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_ei_transforms.py` for a demonstration. Note these require ``kornia`` installed.

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