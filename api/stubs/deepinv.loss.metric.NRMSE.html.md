# NRMSE

### *class* deepinv.loss.metric.NRMSE(method='l2', \*\*kwargs)

Bases: [`NMSE`](https://deepinv.org/api/stubs/deepinv.loss.metric.NMSE.html.md#deepinv.loss.metric.NMSE)

Normalized Root Mean Squared Error metric.

Calculates

$$
\operatorname{NRMSE}(\hat{x},x)
= \frac{\|\hat{x}-x\|_2}{\|x\|_2}
= \sqrt{\operatorname{NMSE}(\hat{x},x)},
$$

where $\hat{x}=\inverse{y}$.

#### NOTE
By default, no reduction is performed in the batch dimension.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import NRMSE
>>> m = NRMSE()
>>> x_net = x = torch.ones(3, 2, 8, 8)
>>> m(x_net, x)
tensor([0., 0., 0.])
```

* **Parameters:**
  * **method** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – normalisation method. Currently only supports `l2`.
  * **complex_abs** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – perform complex magnitude before passing data to metric function.
  * **reduction** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – method used to reduce scores over the batch dimension.
  * **norm_inputs** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – optional input normalization.
  * **center_crop** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *]* *,* *None*) – optional spatial center crop.

<a id="sphx-glr-backref-deepinv-loss-metric-nrmse"></a>

## Examples using `NRMSE`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct an image from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet2d_thumb.png)

[Positron emission tomography (PET) in 2D](https://deepinv.org/auto_examples/physics/demo_pet2d.html.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 2D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This demo shows how to define a non time-of-flight PET scanner, simulate measurements and reconstruct a volume from them.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet3d_thumb.png)

[Positron emission tomography (PET) in 3D](https://deepinv.org/auto_examples/physics/demo_pet3d.html.md)

  <div class="sphx-glr-thumbnail-title">Positron emission tomography (PET) in 3D</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.html.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.html.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div>
<!-- thumbnail-parent-div-close --></div>
