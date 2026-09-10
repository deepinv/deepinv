<a id="physics-functional"></a>

# Functional

The toolbox is based on efficient PyTorch implementations of basic operations such as diagonal multipliers,
Fourier transforms, convolutions, product-convolutions, Radon transform, interpolation mappings.
Similar to the PyTorch structure, they are available within `deepinv.physics.functional`.

```pycon
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
```

#### Functional Routines Overview

| **Function**                                                                                                                                         | **Description**                                                                                                                                  |
|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.physics.functional.conv2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d.html.md#deepinv.physics.functional.conv2d)                               | Performs 2D convolution on input data, commonly used in image processing for filtering and feature extraction.                                   |
| [`deepinv.physics.functional.conv_transpose2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose2d.html.md#deepinv.physics.functional.conv_transpose2d)           | Computes the 2D transposed convolution (deconvolution), used for upsampling or reversing convolutional operations.                               |
| [`deepinv.physics.functional.conv2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv2d_fft.html.md#deepinv.physics.functional.conv2d_fft)                       | Performs 2D convolution using the Fast Fourier Transform (FFT), offering faster performance for large kernel sizes.                              |
| [`deepinv.physics.functional.conv_transpose2d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose2d_fft.html.md#deepinv.physics.functional.conv_transpose2d_fft)   | Computes the 2D transposed convolution with FFT, efficiently implementing upsampling or deconvolution.                                           |
| [`deepinv.physics.functional.conv3d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv3d.html.md#deepinv.physics.functional.conv3d)                               | Performs 3D convolution.                                                                                                                         |
| [`deepinv.physics.functional.conv_transpose3d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose3d.html.md#deepinv.physics.functional.conv_transpose3d)           | Computes the 3D transposed convolution                                                                                                           |
| [`deepinv.physics.functional.conv3d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv3d_fft.html.md#deepinv.physics.functional.conv3d_fft)                       | Performs 3D convolution using FFT, suitable for volumetric data processing in applications like medical imaging.                                 |
| [`deepinv.physics.functional.conv_transpose3d_fft()`](https://deepinv.org/api/stubs/deepinv.physics.functional.conv_transpose3d_fft.html.md#deepinv.physics.functional.conv_transpose3d_fft)   | Computes 3D transposed convolution using FFT, often used for volumetric data reconstruction or upsampling.                                       |
| [`deepinv.physics.functional.product_convolution2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.product_convolution2d.html.md#deepinv.physics.functional.product_convolution2d) | Implements a 2D product convolution, enabling spatially varying convolution across the input image.                                              |
| [`deepinv.physics.functional.multiplier()`](https://deepinv.org/api/stubs/deepinv.physics.functional.multiplier.html.md#deepinv.physics.functional.multiplier)                       | Applies an element-wise multiplier to the input data, typically used to modify pixel intensities or apply masks.                                 |
| [`deepinv.physics.functional.multiplier_adjoint()`](https://deepinv.org/api/stubs/deepinv.physics.functional.multiplier_adjoint.html.md#deepinv.physics.functional.multiplier_adjoint)       | Applies the adjoint of an element-wise multiplier, effectively reversing the scaling applied by `multiplier`.                                    |
| [`deepinv.physics.functional.Radon()`](https://deepinv.org/api/stubs/deepinv.physics.functional.Radon.html.md#deepinv.physics.functional.Radon)                                 | Computes the Radon transform, used in tomography to simulate the projection data from an object.                                                 |
| [`deepinv.physics.functional.IRadon()`](https://deepinv.org/api/stubs/deepinv.physics.functional.IRadon.html.md#deepinv.physics.functional.IRadon)                               | Computes the inverse Radon transform, reconstructing an image from projection data as in CT scan reconstruction.                                 |
| [`deepinv.physics.functional.dct()`](https://deepinv.org/api/stubs/deepinv.physics.functional.dct.html.md#deepinv.physics.functional.dct)                                     | Computes the 1D Discrete Cosine Transform (DCT), commonly used in signal processing and data compression.                                        |
| [`deepinv.physics.functional.idct()`](https://deepinv.org/api/stubs/deepinv.physics.functional.idct.html.md#deepinv.physics.functional.idct)                                   | Computes the inverse 1D Discrete Cosine Transform (IDCT), reconstructing the original signal from its DCT coefficients.                          |
| [`deepinv.physics.functional.dct_2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.dct_2d.html.md#deepinv.physics.functional.dct_2d)                               | Computes the 2D Discrete Cosine Transform (DCT), commonly used in image compression and signal processing.                                       |
| [`deepinv.physics.functional.idct_2d()`](https://deepinv.org/api/stubs/deepinv.physics.functional.idct_2d.html.md#deepinv.physics.functional.idct_2d)                             | Computes the inverse 2D Discrete Cosine Transform (IDCT), reconstructing the original image from its DCT coefficients.                           |
| [`deepinv.physics.functional.XrayTransform()`](https://deepinv.org/api/stubs/deepinv.physics.functional.XrayTransform.html.md#deepinv.physics.functional.XrayTransform)                 | X-ray Transform operator with `astra-toolbox` backend. Computes forward projection and backprojection used in CT reconstruction.                 |
| [`deepinv.physics.functional.histogramdd()`](https://deepinv.org/api/stubs/deepinv.physics.functional.histogramdd.html.md#deepinv.physics.functional.histogramdd)                     | Computes the histogram of a multi-dimensional dataset, useful in statistical analysis and data visualization.                                    |
| [`deepinv.physics.functional.histogram()`](https://deepinv.org/api/stubs/deepinv.physics.functional.histogram.html.md#deepinv.physics.functional.histogram)                         | Computes the histogram of 1D or 2D data, often used for intensity distribution analysis in image processing.                                     |
| [`deepinv.physics.functional.imresize_matlab()`](https://deepinv.org/api/stubs/deepinv.physics.functional.imresize_matlab.html.md#deepinv.physics.functional.imresize_matlab)             | MATLAB bicubic imresize function implemented in PyTorch.                                                                                         |
| [`deepinv.physics.functional.power_method()`](https://deepinv.org/api/stubs/deepinv.physics.functional.power_method.html.md#deepinv.physics.functional.power_method)                   | Implements the power method to compute the largest singular value of a linear operator defined by forward and adjoint functions.                 |
| [`deepinv.physics.functional.random_choice()`](https://deepinv.org/api/stubs/deepinv.physics.functional.random_choice.html.md#deepinv.physics.functional.random_choice)                 | Randomly selects elements from a given input tensor based on specified probabilities, useful for stochastic sampling in various applications.    |
| [`deepinv.physics.functional.gaussian_blur()`](https://deepinv.org/api/stubs/deepinv.physics.functional.gaussian_blur.html.md#deepinv.physics.functional.gaussian_blur)                 | Generates a Gaussian blur kernel in 1D, 2D or 3D, commonly used to model point spread functions.                                                 |
| [`deepinv.physics.functional.bilinear_filter()`](https://deepinv.org/api/stubs/deepinv.physics.functional.bilinear_filter.html.md#deepinv.physics.functional.bilinear_filter)             | Generates a bilinear filter kernel, often used for image resizing and interpolation.                                                             |
| [`deepinv.physics.functional.bicubic_filter()`](https://deepinv.org/api/stubs/deepinv.physics.functional.bicubic_filter.html.md#deepinv.physics.functional.bicubic_filter)               | Generates a bicubic filter kernel, providing smoother results than bilinear filtering for image resizing.                                        |
| [`deepinv.physics.functional.sinc_filter()`](https://deepinv.org/api/stubs/deepinv.physics.functional.sinc_filter.html.md#deepinv.physics.functional.sinc_filter)                     | Generates a sinc filter kernel, used for ideal low-pass filtering in signal processing and image resampling.                                     |
| [`deepinv.physics.functional.liu_jia_pad()`](https://deepinv.org/api/stubs/deepinv.physics.functional.liu_jia_pad.html.md#deepinv.physics.functional.liu_jia_pad)                     | Pads an image to make it have smooth circular boundaries for use in spectral deconvolution, reducing ringing artifacts in the deblurred outputs. |
| [`deepinv.physics.split_measurements()`](https://deepinv.org/api/stubs/deepinv.physics.split_measurements.html.md#deepinv.physics.split_measurements)                             | Splits tomography measurements into interleaved angular / vector subsets.                                                                        |
| [`deepinv.physics.split_physics()`](https://deepinv.org/api/stubs/deepinv.physics.split_physics.html.md#deepinv.physics.split_physics)                                       | Builds stacked tomography physics with one operator per interleaved angular / vector subset.                                                     |
