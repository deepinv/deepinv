# idct

### deepinv.physics.functional.idct(x, norm=None)

The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

For the meaning of the parameter `norm`, see:
[https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html)

Parts of this code are adapted from the `torch-dct` repository by zh217: [https://github.com/zh217/torch-dct](https://github.com/zh217/torch-dct)

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the input signal
  * **norm** (*None* *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the normalization, `None` or `'ortho'`
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the inverse DCT-II of the signal over the last dimension
