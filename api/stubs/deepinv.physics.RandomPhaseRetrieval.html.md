# RandomPhaseRetrieval

### *class* deepinv.physics.RandomPhaseRetrieval(m, img_size, channelwise=False, dtype=torch.cfloat, device='cpu', rng=None, \*\*kwargs)

Bases: [`PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval)

Random Phase Retrieval forward operator. Creates a random $m \times n$ sampling matrix $B$ where $n$ is the number of elements of the signal and $m$ is the number of measurements.

This class generates a random i.i.d. Gaussian matrix

$$
B_{i,j} \sim \mathcal{N} \left( 0, \frac{1}{2m} \right) + \mathrm{i} \mathcal{N} \left( 0, \frac{1}{2m} \right).
$$

An existing operator can be loaded from a saved .pth file via `self.load_state_dict(save_path)`, in a similar fashion to [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

* **Parameters:**
  * **m** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of measurements.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – shape (C, H, W) of inputs.
  * **channelwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – Channels are processed independently using the same random forward operator.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Forward matrix is stored as a dtype. Default is torch.cfloat.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to store the forward matrix.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – (optional) a pseudorandom random number generator for the parameter generation.
    If `None`, the default Generator of PyTorch will be used.

<hr />

* **Examples:**
  Random phase retrieval operator with 10 measurements for a 3x3 image:
  ```pycon
  >>> from deepinv.physics import RandomPhaseRetrieval
  >>> seed = torch.manual_seed(0) # Random seed for reproducibility
  >>> x = torch.randn((1, 1, 3, 3),dtype=torch.cfloat) # Define random 3x3 image
  >>> physics = RandomPhaseRetrieval(m=6, img_size=(1, 3, 3), rng=torch.Generator('cpu'))
  >>> physics(x)
  tensor([[3.8405, 2.2588, 0.0146, 3.0864, 1.8075, 0.1518]])
  ```

<a id="sphx-glr-backref-deepinv-physics-randomphaseretrieval"></a>

## Examples using `RandomPhaseRetrieval`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a random phase retrieval operator and generate phaseless measurements from a given image. The example showcases 4 different reconstruction methods to recover the image from the phaseless measurements:">  <div class="sphx-glr-thumbnail-title">Random phase retrieval and reconstruction methods.</div>
</div>
<!-- thumbnail-parent-div-close --></div>
