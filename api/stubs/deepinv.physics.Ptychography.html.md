# Ptychography

### *class* deepinv.physics.Ptychography(img_size=None, probe=None, shifts=None, device='cpu', \*\*kwargs)

Bases: [`PhaseRetrieval`](https://deepinv.org/api/stubs/deepinv.physics.PhaseRetrieval.html.md#deepinv.physics.PhaseRetrieval)

Ptychography forward operator.

Corresponding to the operator

$$
\forw{x} = \left| Bx \right|^2
$$

where $B$ is the linear forward operator defined by a [`deepinv.physics.PtychographyLinearOperator`](https://deepinv.org/api/stubs/deepinv.physics.PtychographyLinearOperator.html.md#deepinv.physics.PtychographyLinearOperator) object.

* **Parameters:**
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Shape of the input image.
  * **probe** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – A tensor of shape `img_size` representing the probe function.
    If None, a disk probe is generated with `deepinv.physics.phase_retrieval.build_probe` function.
  * **shifts** (*None* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – A 2D array of shape (`n_img`, 2) corresponding to the shifts for the probe.
    If None, shifts are generated with `deepinv.physics.phase_retrieval.generate_shifts` function.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device “cpu” or “gpu”.

<hr />

* **Examples:**

```pycon
>>> from deepinv.physics import Ptychography
>>> import torch
>>> img_size = (1, 64, 64)  # input image
>>> physics = Ptychography(img_size=img_size)
>>> x = torch.randn(img_size, dtype=torch.cfloat)
>>> y = physics(x)  # Apply the Ptychography forward operator
>>> print(y.shape) # 25 probe positions by default
torch.Size([1, 25, 64, 64])
```

<a id="sphx-glr-backref-deepinv-physics-ptychography"></a>

## Examples using `Ptychography`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to create a Ptychography phase retrieval operator and generate phaseless measurements from a given image.">  <div class="sphx-glr-thumbnail-title">Ptychography phase retrieval</div>
</div>
<!-- thumbnail-parent-div-close --></div>
