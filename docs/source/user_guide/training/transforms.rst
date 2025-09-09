.. _transform:

Transforms
=================

This module contains different transforms which can be used for data augmentation or together with the equivariant losses.

We implement various geometric transforms, ranging from Euclidean to homography and diffeomorphisms, some of which offer group-theoretic properties.

**See** :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_transforms.py` **for example usage and visualisations.**

Transforms inherit from :class:`deepinv.transform.Transform`. Transforms can also be stacked by summing them, chained by multiplying them (i.e. product group), or joined via ``|`` to randomly select.
There are numerous other parameters e.g to randomly transform multiple times at once, to constrain the parameters to a range etc.

Common usages of transforms:

- | Make a denoiser equivariant using :class:`deepinv.models.EquivariantDenoiser`
  | by performing Reynolds averaging using ``symmetrize()``. See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_ei_transforms.py`.
- | Equivariant imaging (EI) using the :class:`deepinv.loss.EILoss` loss.
  | See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_equivariant_imaging.py`.

If needed, transforms can also be made deterministic by passing in specified parameters to the forward method.
This allows every transform to have its own deterministic inverse using ``transform.inverse()``.
Transforms can also be seamlessly integrated with existing ``torchvision`` transforms and can also accept video (5D) input.


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
We provide the following simple geometric transforms.

.. list-table:: Simple Transformations
   :header-rows: 1

   * - **Transform**
     - **Uses Interpolation**
     - **Exact Inversion**

   * - :class:`deepinv.transform.Rotate`
     - Yes
     - No

   * - :class:`deepinv.transform.Shift`
     - No
     - Yes

   * - :class:`deepinv.transform.Scale`
     - Yes
     - No

   * - :class:`deepinv.transform.Reflect`
     - No
     - Yes

   * - :class:`deepinv.transform.Identity`
     - No
     - Yes

Advanced transforms
-------------------

We implement the following further geometric transforms.
The projective transformations formulate the image transformations using the pinhole camera model,
from which various transformation subgroups can be derived.
See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_ei_transforms.py` for a demonstration.
Note these require installing the library ``kornia``.

.. list-table:: Advanced Transformations
   :header-rows: 1

   * - **Transform**
     - **Description**

   * - :class:`deepinv.transform.Homography`
     - A general projective transformation allowing perspective distortion and transformation between different planes.

   * - :class:`deepinv.transform.projective.Euclidean`
     - A rigid transformation that preserves angles and distances, allowing only rotation and translation.

   * - :class:`deepinv.transform.projective.Similarity`
     - A transformation that preserves shapes through scaling, rotation, and translation, maintaining proportions.

   * - :class:`deepinv.transform.projective.Affine`
     - A transformation preserving parallel lines, allowing scaling, rotation, translation, and shearing.

   * - :class:`deepinv.transform.projective.PanTiltRotate`
     - A specialized transformation that simulates pan, tilt, and rotation effects in imaging.

   * - :class:`deepinv.transform.CPABDiffeomorphism`
     - A continuous piecewise affine transformation allowing for smooth and invertible deformations across an image.


Video transforms
----------------

While all geometric transforms accept video input, the following transforms work specifically in the time dimension.
These can be easily compounded with geometric transformations using the ``*`` operation.

.. list-table:: Time Transforms
   :header-rows: 1

   * - **Transform**
     - **Description**

   * - :class:`deepinv.transform.ShiftTime`
     - A temporal shift in the time dimension.

Non-geometric transforms
------------------------

Non-geometric transforms are often used for data augmentation.
Note that not all of these are necessarily invertible or form groups.

.. list-table:: Non-geometric Transforms
   :header-rows: 1

   * - **Transform**
     - **Description**

   * - :class:`deepinv.transform.RandomNoise`
     - Add random noise to data (non-invertible).
   * - :class:`deepinv.transform.RandomPhaseError`
     - Add random phase error to frequency data.