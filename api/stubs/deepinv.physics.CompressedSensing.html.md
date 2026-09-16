# CompressedSensing

### *class* deepinv.physics.CompressedSensing(m, img_size, channelwise=False, dtype=torch.float, device='cpu', rng=None, \*\*kwargs)

Bases: [`LinearPhysics`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics)

Compressed Sensing forward operator. Creates a random sampling $m \times n$ matrix where $n$ is the
number of elements of the signal, i.e., `np.prod(img_size)` and `m` is the number of measurements.

This class generates a random iid Gaussian matrix

$$
A_{i,j} \sim \mathcal{N}(0,\frac{1}{m})
$$

For image sizes bigger than 32 x 32, the forward computation can be prohibitively expensive due to its $O(mn)$ complexity.
In this case, we recommend using [`deepinv.physics.StructuredRandom`](https://deepinv.org/api/stubs/deepinv.physics.StructuredRandom.html.md#deepinv.physics.StructuredRandom) instead.

An existing operator can be loaded from a saved .pth file via `self.load_state_dict(save_path)`,
in a similar fashion to [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

#### NOTE
The forward operator has a norm which tends to $(1+\sqrt{n/m})^2$ for large $n$
and $m$ due to the [Marcenko-Pastur law](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution).

If `dtype=torch.cfloat`, the forward operator will be generated as a random i.i.d. complex Gaussian matrix

$$
A_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).
$$

* **Parameters:**
  * **m** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of measurements.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape (C, H, W) of inputs.
  * **channelwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Channels are processed independently using the same random forward operator.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Forward matrix is stored as a dtype. For complex matrices, use torch.cfloat. Default is torch.float.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to store the forward matrix.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – (optional) a pseudorandom random number generator for the parameter generation.
    If `None`, the default Generator of PyTorch will be used.

<hr />

* **Examples:**
  Compressed sensing operator with 100 measurements for a 3x3 image:
  ```pycon
  >>> from deepinv.physics import CompressedSensing
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn(1, 1, 3, 3) # Define random 3x3 image
  >>> physics = CompressedSensing(m=10, img_size=(1, 3, 3), rng=torch.Generator('cpu'))
  >>> physics(x)
  tensor([[-1.7769,  0.6160, -0.8181, -0.5282, -1.2197,  0.9332, -0.1668,  1.5779,
            0.6752, -1.5684]])
  ```

<a id="sphx-glr-backref-deepinv-physics-compressedsensing"></a>

## Examples using `CompressedSensing`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div>
<!-- thumbnail-parent-div-close --></div>
