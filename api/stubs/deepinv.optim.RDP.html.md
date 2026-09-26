# RDP

### *class* deepinv.optim.RDP(gamma=2.0, \*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Relative Difference Prior (RDP).

This prior was proposed for emission tomography by Nuyts *et al.*<sup>[1](#footcite-nuytsconcavepriorpenalizing2002)</sup>.
It favors sharp transitions in non-negative images and is particularly useful when the signal has a large amplitude.
It penalizes relative rather than absolute differences between neighboring voxels:

$$
\reg{x} = \sum_{\{j,k\} \in \mathcal{N}} \frac{(x_j-x_k)^2}{x_j+x_k+\gamma |x_j-x_k|},
$$

where $\mathcal{N}$ contains each axis-adjacent spatial pair once.
The batch and channel axes are not included in the neighborhood.

#### WARNING
Negative values in the image can make the denominator cancel.
This implementation is only valid for non-negative images.

* **Parameters:**
  **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – edge-preservation parameter $\gamma$. Larger values reduce the penalty on large relative differences. Default: `2.0`.

<hr />

* **References:**

* <a id='footcite-nuytsconcavepriorpenalizing2002'>**[1]**</a> Johan Nuyts, Dirk Bequé, Patrick Dupont, and Luc Mortelmans. A concave prior penalizing relative differences for maximum-a-posteriori reconstruction in emission tomography. *IEEE Transactions on Nuclear Science*, 49(1):56–60, February 2002. [doi:10.1109/TNS.2002.998681](https://doi.org/10.1109/TNS.2002.998681).

#### fn(x, \*args, \*\*kwargs)

Compute the Relative Difference Prior at $x$.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – non-negative image or volume.
* **Returns:**
  prior value for each element of the batch.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### grad(x, \*args, \*\*kwargs)

Compute the gradient of the Relative Difference Prior at $x$.

The zero gradient is selected for pairs in which both voxels are zero.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – non-negative image or volume.
* **Returns:**
  gradient with the same shape as $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-rdp"></a>

## Examples using `RDP`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.html.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.html.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div>
<!-- thumbnail-parent-div-close --></div>
