# AugmentConsistencyLoss

### *class* deepinv.loss.AugmentConsistencyLoss(T_i=None, T_e=None, metric=None, no_grad=True, rng=None, \*args, \*\*kwargs)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Data augmentation consistency (DAC) loss.

Performs data augmentation in measurement domain as proposed by Desai *et al.*<sup>[1](#footcite-desai2021vortex)</sup>.

The loss is defined as follows:

$\mathcal{L}(T_e\inverse{y,A},\inverse{T_i y,A T_e^{-1}})$

where $T_i$ is a [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) for which we should learn an invariant mapping,
and $T_e$ is a [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) for which we should learn an equivariant mapping.

#### NOTE
If $T_e$ is specified, the mapping is performed in the image domain and the model is assumed to take $A^\top y$ as input.

By default, for $T_i$ we add random noise [`deepinv.transform.RandomNoise`](https://deepinv.org/api/stubs/deepinv.transform.RandomNoise.html.md#deepinv.transform.RandomNoise) and random phase error [`deepinv.transform.RandomPhaseError`](https://deepinv.org/api/stubs/deepinv.transform.RandomPhaseError.html.md#deepinv.transform.RandomPhaseError).
By default, for $T_e$ we use random shift [`deepinv.transform.Shift`](https://deepinv.org/api/stubs/deepinv.transform.Shift.html.md#deepinv.transform.Shift) and random rotates [`deepinv.transform.Rotate`](https://deepinv.org/api/stubs/deepinv.transform.Rotate.html.md#deepinv.transform.Rotate).

#### NOTE
See [Transforms](https://deepinv.org/user_guide/training/transforms.html.md#transform) for a guide on all available transforms, and how to compose them. For example, you can easily
compose further transforms such as  `Rotate(rng=rng, multiples=90) | Scale(factors=[0.75, 1.25], rng=rng) | Reflect(rng=rng)`.

* **Parameters:**
  * **T_i** ([*deepinv.transform.Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – invariant transform performed on $y$.
  * **T_e** ([*deepinv.transform.Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform)) – equivariant transform performed on $A^\top y$.
  * **metric** ([*deepinv.loss.metric.Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric for calculating loss.
  * **no_grad** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, only propagate gradients through augmented branch as per original paper,
    if `False`, propagate through both branches.
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – torch random number generator to pass to transforms.

<hr />

* **References:**

* <a id='footcite-desai2021vortex'>**[1]**</a> Arjun D Desai, Beliz Gunel, Batu Ozturkler, Harris Beg, Shreyas Vasanawala, Brian Hargreaves, Christopher Re, John M Pauly, and Akshay Chaudhari. Vortex: physics-driven data augmentations using consistency training for robust accelerated mri reconstruction. In *Medical Imaging with Deep Learning*. 2021.

#### forward(x_net, y, physics, model, \*\*kwargs)

Data augmentation consistency loss forward pass.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Reconstructed image $\inverse{y}$.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurement.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss, the tensor size might be (1,) or (batch size,).
