# RecoveryCoefficient

### *class* deepinv.loss.metric.RecoveryCoefficient(eps=None, \*\*kwargs)

Bases: [`Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.md#deepinv.loss.metric.Metric)

Recovery Coefficient metric used in emission tomography.

Computes the ratio between the total reconstructed activity and the total
ground-truth activity inside a given region of interest, defined by a mask:

$$
RC = \frac{\sum_i \hat{x}_i m_i}
          {\sum_i x_i m_i + \varepsilon}
$$

where $\hat{x}$ is the reconstructed image, $x$ is the ground-truth
image, $m$ is a binary or weighted mask defining the region of interest,
and $\varepsilon$ is a small constant added for numerical stability.

A value of `1` indicates perfect recovery of activity within the masked region.
Values below `1` indicate underestimation, while values above `1` indicate
overestimation.

#### NOTE
This metric requires a `mask` keyword argument to be provided during
evaluation.

#### NOTE
Higher values are better when they are closer to `1`. Therefore,
`lower_better=False`.

* **Example:**

```pycon
>>> import torch
>>> from deepinv.loss.metric import RecoveryCoefficient
>>> m = RecoveryCoefficient()
>>> x = torch.ones(2, 1, 4, 4)
>>> x_net = torch.ones(2, 1, 4, 4)
>>> mask = torch.ones_like(x)
>>> m(x_net, x, mask=mask)
tensor([1., 1.])
```

* **Parameters:**
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Small constant added to the denominator for numerical stability.
  * **kwargs** – Additional keyword arguments passed to the parent
    [`deepinv.loss.metric.Metric`](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.md#deepinv.loss.metric.Metric) class.

#### invert_metric(m)

“Invert metric for use as a training loss.
Recovery Coefficient is optimal at 1 so a sign flip does not produce a valid
loss
:param torch.Tensor m: calculated metric

<a id="sphx-glr-backref-deepinv-loss-metric-recoverycoefficient"></a>

## Examples using `RecoveryCoefficient`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a 2D slice from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. The slice contains five hot lesions, we simulate a sinogram with deepinv.physics.PET and we compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_2d_thumb.png)

[2D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_2d.md)

  <div class="sphx-glr-thumbnail-title">2D PET reconstruction with the Brainweb dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example reconstructs a volume from the BrainWeb \`&lt;https://github.com/casperdcl/brainweb&gt;\`_ positron emission tomography (PET) dataset. We compare standard PET reconstruction algorithms with methods that support penalized objective functions.">![](auto_examples/physics/images/thumb/sphx_glr_demo_pet_brainweb_3d_thumb.png)

[3D PET reconstruction with the Brainweb dataset](https://deepinv.org/auto_examples/physics/demo_pet_brainweb_3d.md)

  <div class="sphx-glr-thumbnail-title">3D PET reconstruction with the Brainweb dataset</div>
</div>
<!-- thumbnail-parent-div-close --></div>
