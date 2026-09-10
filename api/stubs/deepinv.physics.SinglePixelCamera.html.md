# SinglePixelCamera

### *class* deepinv.physics.SinglePixelCamera(m, img_size, fast=True, ordering='sequency', device='cpu', dtype=torch.float32, rng=None, \*\*kwargs)

Bases: [`DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics)

Single pixel imaging camera.

Linear imaging operator with binary entries.

If `fast=True`, the operator uses a 2D subsampled Hadamard transform, which keeps the first $m$ modes
according to the `ordering` parameter, set by default to [sequency ordering](https://en.wikipedia.org/wiki/Walsh_matrix#Sequency_ordering).
In this case, the images should have a size which is a power of 2.

If `fast=False`, the operator is a random iid binary matrix with equal probability of $1/\sqrt{m}$ or
$-1/\sqrt{m}$.

Both options allow for an efficient singular value decomposition (see [`deepinv.physics.DecomposablePhysics`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics))
The operator is always applied independently across channels.

It is recommended to use `fast=True` for image sizes bigger than 32 x 32, since the forward computation with
`fast=False` has an $O(mn)$ complexity, whereas with `fast=True` it has an $O(n \log n)$ complexity.

An existing operator can be loaded from a saved `.pth` file via `self.load_state_dict(save_path)`,
in a similar fashion to [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

* **Parameters:**
  * **m** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of single pixel measurements per acquisition (m).
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape (C, H, W) of images.
  * **fast** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – The operator is iid binary if false, otherwise A is a 2D subsampled hadamard transform.
  * **ordering** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – The ordering of selecting the first m measurements, available options are: `'sequency'`, `'cake_cutting'`, `'zig_zag'`, `'xy'`.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – (optional) a pseudorandom random number generator for the parameter generation.
    If `None`, the default Generator of PyTorch will be used.

<hr />

* **Examples:**
  SinglePixelCamera operators with 16 binary patterns for 32x32 image:
  ```pycon
  >>> from deepinv.physics import SinglePixelCamera
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn((1, 1, 32, 32)) # Define random 32x32 image
  >>> physics = SinglePixelCamera(m=16, img_size=(1, 32, 32), fast=True)
  >>> torch.sum(physics.mask).item() # Number of measurements
  16.0
  >>> torch.round(physics(x)[:, :, :3, :3]).abs() # Compute measurements
  tensor([[[[1., 0., 1.],
            [0., 0., 0.],
            [0., 0., 0.]]]])
  ```

<a id="sphx-glr-backref-deepinv-physics-singlepixelcamera"></a>

## Examples using `SinglePixelCamera`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of some of the forward operators implemented in DeepInverse. We restrict ourselves to operators where the signal is a 2D image. The full list of operators can be found in physics.">  <div class="sphx-glr-thumbnail-title">Tour of forward sensing operators</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo illustrates the impact of different Hadamard pattern ordering algorithms in the Single Pixel Camera (SPC), a computational imaging system that uses a single photodetector to capture images by projecting the scene onto a series of patterns. The SPC is implemented in the DeepInverse library.">  <div class="sphx-glr-thumbnail-title">Pattern Ordering in a Compressive Single Pixel Camera</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to define your own optimization algorithm. For example, here, we implement the Primal-Dual Condat-Vu (CV) algorithm, and apply it for Single Pixel Camera reconstruction.">  <div class="sphx-glr-thumbnail-title">PnP with custom optimization algorithm (Primal-Dual Condat-Vu)</div>
</div>
<!-- thumbnail-parent-div-close --></div>
