.. _physics_functional:

Functional
--------------------

The toolbox is based on efficient PyTorch implementations of basic operations such as diagonal multipliers, Fourier transforms, convolutions, product-convolutions, Radon transform, interpolation mappings.
Similar to the PyTorch structure, they are available within :py:mod:`deepinv.physics.functional`.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.physics.functional.conv2d
   deepinv.physics.functional.conv_transpose2d
   deepinv.physics.functional.conv2d_fft
   deepinv.physics.functional.conv_transpose2d_fft
   deepinv.physics.functional.conv3d_fft
   deepinv.physics.functional.conv_transpose3d_fft
   deepinv.physics.functional.product_convolution2d
   deepinv.physics.functional.multiplier
   deepinv.physics.functional.multiplier_adjoint
   deepinv.physics.functional.Radon
   deepinv.physics.functional.IRadon
   deepinv.physics.functional.histogramdd
   deepinv.physics.functional.histogram

.. doctest::

    >>> import torch
    >>> import deepinv as dinv

    >>> x = torch.zeros((1, 1, 16, 16)) # Define black image of size 16x16
    >>> x[:, :, 8, 8] = 1 # Define one white pixel in the middle
    >>> filter = torch.ones((1, 1, 3, 3)) / 4
    >>>
    >>> padding = "circular"
    >>> Ax = dinv.physics.functional.conv2d(x, filter, padding)
    >>> print(Ax[:, :, 7:10, 7:10])
    tensor([[[[0.2500, 0.2500, 0.2500],
              [0.2500, 0.2500, 0.2500],
              [0.2500, 0.2500, 0.2500]]]])
    >>>
    >>> _ = torch.manual_seed(0)
    >>> y = torch.randn_like(Ax)
    >>> z = dinv.physics.functional.conv_transpose2d(y, filter, padding)
    >>> print((Ax * y).sum(dim=(1, 2, 3)) - (x * z).sum(dim=(1, 2, 3)))
    tensor([5.9605e-08])