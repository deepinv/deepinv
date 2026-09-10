# GaussianBlurGenerator

### *class* deepinv.physics.generator.GaussianBlurGenerator(psf_size, sigma_min=0.5, sigma_max=5.0, isotropic=True, angle_min=0.0, angle_max=360.0, rng=None, device='cpu', dtype=torch.float32)

Bases: [`PSFGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.PSFGenerator.html.md#deepinv.physics.generator.PSFGenerator)

Random Gaussian blur generator. Generates 1D, 2D, or 3D Gaussian kernels with random standard deviations and rotation angles.

* **Parameters:**
  * **psf_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – the shape of the generated point spread function (PSF). The dimension (1D, 2D, or 3D) of the kernel is determined by the length of the `psf_size` tuple.
  * **sigma_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]*) – the minimum standard deviation(s) for the Gaussian kernel. If a single value is provided, it is applied to all dimensions. If a tuple is provided, it should have the same length as the number of dimensions and specify the minimum sigma for each dimension.
  * **sigma_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]*) – the maximum standard deviation(s) for the Gaussian kernel. Follows the same format as `sigma_min`.
  * **isotropic** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, the generated Gaussian kernels will be isotropic (same sigma for all dimensions). If False, the kernels can be anisotropic (different sigma for each dimension). Defaults to True.
  * **angle_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]*) – the minimum rotation angle(s) for the Gaussian kernel in degrees. For 2D kernels, this is a single angle of rotation in the plane. For 3D kernels, this can be a tuple of three angles (alpha, beta, gamma) representing minimum rotation values around the x, y, and z axes respectively. In 3D, if a single angle is provided, it is used as minimum value for all axes.
  * **angle_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *|* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*float*](https://docs.python.org/3.9/library/functions.html#float) *,*  *...* *]*) – the maximum rotation angle(s) for the Gaussian kernel in degrees. Follows the same format as `angle_min`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – PyTorch random number generator for reproducibility. If `None`, a torch.Generator will be created on the specified device.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the device to create the tensors on. Defaults to “cpu”.
  * **dtype** ([*type*](https://docs.python.org/3.9/library/functions.html#type)) – the data type of the generated tensors. Defaults to torch.float32.

#### NOTE
Always generates single-channel PSFs of shape `(B, 1, H, W)`.

<hr />

* **Examples:**

```pycon
>>> import deepinv as dinv
>>> rng = torch.Generator(device="cpu").manual_seed(0)
>>> generator = dinv.physics.generator.GaussianBlurGenerator((7, 7), device="cpu", rng=rng, isotropic=False)
>>> params = generator.step(batch_size=4)  # dict_keys(['filter'])
>>> print(params['filter'].shape)
torch.Size([4, 1, 7, 7])
>>> dinv.utils.plot(params['filter'])
```

![image](../../_build/plot_directive/api/stubs/deepinv-physics-generator-GaussianBlurGenerator-1.*)
