.. _physics_functional:

Functional
-----------

The toolbox is based on efficient PyTorch implementations of basic operations such as diagonal multipliers,
Fourier transforms, convolutions, product-convolutions, Radon transform, interpolation mappings.
Similar to the PyTorch structure, they are available within ``deepinv.physics.functional``.

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


.. list-table:: Functional Routines Overview
   :header-rows: 1

   * - **Function**
     - **Description**

   * - :func:`deepinv.physics.functional.conv2d`
     - Performs 2D convolution on input data, commonly used in image processing for filtering and feature extraction.

   * - :func:`deepinv.physics.functional.conv_transpose2d`
     - Computes the 2D transposed convolution (deconvolution), used for upsampling or reversing convolutional operations.

   * - :func:`deepinv.physics.functional.conv2d_fft`
     - Performs 2D convolution using the Fast Fourier Transform (FFT), offering faster performance for large kernel sizes.

   * - :func:`deepinv.physics.functional.conv_transpose2d_fft`
     - Computes the 2D transposed convolution with FFT, efficiently implementing upsampling or deconvolution.

   * - :func:`deepinv.physics.functional.conv3d_fft`
     - Performs 3D convolution using FFT, suitable for volumetric data processing in applications like medical imaging.

   * - :func:`deepinv.physics.functional.conv_transpose3d_fft`
     - Computes 3D transposed convolution using FFT, often used for volumetric data reconstruction or upsampling.

   * - :func:`deepinv.physics.functional.product_convolution2d`
     - Implements a 2D product convolution, enabling spatially varying convolution across the input image.

   * - :func:`deepinv.physics.functional.multiplier`
     - Applies an element-wise multiplier to the input data, typically used to modify pixel intensities or apply masks.

   * - :func:`deepinv.physics.functional.multiplier_adjoint`
     - Applies the adjoint of an element-wise multiplier, effectively reversing the scaling applied by `multiplier`.

   * - :func:`deepinv.physics.functional.Radon`
     - Computes the Radon transform, used in tomography to simulate the projection data from an object.

   * - :func:`deepinv.physics.functional.IRadon`
     - Computes the inverse Radon transform, reconstructing an image from projection data as in CT scan reconstruction.
  
   * - :func:`deepinv.physics.functional.XrayTransform`
     - X-ray Transform operator with ``astra-toolbox`` backend. Computes forward projection and backprojection used in CT reconstruction.

   * - :func:`deepinv.physics.functional.histogramdd`
     - Computes the histogram of a multi-dimensional dataset, useful in statistical analysis and data visualization.

   * - :func:`deepinv.physics.functional.histogram`
     - Computes the histogram of 1D or 2D data, often used for intensity distribution analysis in image processing.

   * - :func:`deepinv.physics.functional.imresize_matlab`
     - MATLAB bicubic imresize function implemented in PyTorch.