# imresize_matlab

### deepinv.physics.functional.imresize_matlab(x, scale=None, sizes=None, kernel='cubic', sigma=2, padding_type='reflect', antialiasing=True)

MATLAB imresize reimplementation.

A standalone PyTorch implementation for fast and efficient bicubic resampling.
The resulting values are the same to MATLAB function imresize(‘bicubic’) with `reflect` padding.

Code reproduced with modifications from [https://github.com/sanghyun-son/bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape (H,W), (C,H,W) or (B,C,H,W)
  * **scale** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – imresize scale factor. > 1 = upsample.
  * **sizes** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – optional output image size following MATLAB convention.
  * **kernel** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – downsampling kernel, choose between ‘cubic’ (for MATLAB bicubic) or ‘gaussian’.
  * **sigma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Gaussian kernel size. Ignored if kernel is not gaussian.
  * **padding_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – reflect padding only.
  * **antialiasing** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to perform antialiasing.
* **Returns:**
  torch.Tensor: output resized image.
* **Example:**
  ```pycon
  >>> import torch
  >>> from deepinv.physics.functional import imresize_matlab
  >>> x = torch.randn(1, 1, 8, 8)
  >>> y = imresize_matlab(x, scale=2)
  >>> y.shape
  torch.Size([1, 1, 16, 16])
  ```
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
