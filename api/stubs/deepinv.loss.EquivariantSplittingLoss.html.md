# EquivariantSplittingLoss

### *class* deepinv.loss.EquivariantSplittingLoss(, mask_generator=None, consistency_loss=None, prediction_loss=None, eval_n_samples=5, transform=None, eval_transform=None, img_size=None)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Equivariant splitting loss.

Implements the measurement splitting loss proposed by Sechaud *et al.*<sup>[1](#footcite-sechaud26equivariant)</sup>. It generalizes the regular [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss) by providing an additional measurement consistency term supporting noise-less losses like [`deepinv.loss.MCLoss`](https://deepinv.org/api/stubs/deepinv.loss.MCLoss.html.md#deepinv.loss.MCLoss), but also noise-aware losses including [`deepinv.loss.R2RLoss`](https://deepinv.org/api/stubs/deepinv.loss.R2RLoss.html.md#deepinv.loss.R2RLoss) and [`deepinv.loss.SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.html.md#deepinv.loss.SureGaussianLoss). Moreover, it automatically renders the base reconstructor equivariant using the Reynolds averaging implemented in [`deepinv.models.EquivariantReconstructor`](https://deepinv.org/api/stubs/deepinv.models.EquivariantReconstructor.html.md#deepinv.models.EquivariantReconstructor).

The training loss takes the general form:

$$
\mathcal{L}_{\mathrm{ES}} (y, A, f) = \mathbb{E}_g \Big\{ \mathbb{E}_{y_1, A_1 \mid y, A T_g} \Big\{ \underbrace{\| A_1 R(y_1, A_1) - A_1 x \|^2}_{\text{Consistency term}} + \underbrace{\| A_2 R(y_1, A_1) - A_2 x \|^2}_{\text{Prediction term}} \Big\} \Big\}
$$

where $R$ denotes the reconstructor, $A$ the physics operator, $x$ the ground truth image, $y$ the measurement, $T_g$ a group action (e.g., rotations).

The second expectation is taken over the distribution specified by `mask_generator` of all possible splittings of $A T_g$, i.e., $A T_g = [A_1^\top, A_2^\top]^\top$, with the associated measurements denoted as $y_1$ and $y_2$.

The main idea behind equivariant splitting is that the more the reconstructor is equivariant to suitable transformations, the better the final performance will be. A general way to make a reconstructor $\tilde{R}$ equivariant is to add a group averaging step in the reconstructor,

$$
R(y, A) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g \tilde{R}(y, A T_g)
$$

which is generally estimated using a Monte Carlo approach at training time. For this reason, [`EquivariantSplittingLoss`](#deepinv.loss.EquivariantSplittingLoss) takes two different instances of [`deepinv.transform.Transform`](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) as input: one for training `transform` and one for evaluation `eval_transform`.

It is also possible to design an equivariant reconstructor without Reynolds averaging, using equivariant layers. In that case, Reynolds averaging can be disabled to avoid its additional computational cost by leaving `transform` and `eval_transform` to `None`.

The training loss consists in two terms, a consistency term where the comparison is performed against $A_1 x$ and a prediction term where the comparison is performed against $A_2 x$. Two parameters control the way these two terms are computed: `consistency_loss` and `prediction_loss`.

In the absence of noise, the equivariant splitting loss $\mathcal{L}_{\mathrm{ES}}$ can be computed exactly without having access to ground truth images. Indeed, in that case, $A_1 x = y_1$ and $A_2 x = y_2$. Setting `consistency_loss` and `prediction_loss` to `deepinv.loss.MCLoss(metric=deepinv.metric.MSE())` allows to compute the loss this way.

In the presence of noise, as long as the splitting scheme is chosen so that the resulting noise components are independent, the prediction term can be estimated without bias using `deepinv.loss.MCLoss(metric=deepinv.metric.MSE())` for `prediction_loss`. This is notably the case for typical splitting schemes, e.g., [`deepinv.physics.generator.BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) when the noise is pixel-wise independent, e.g., [`deepinv.physics.GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise).

The consistency term should be set to one of the self-supervised denoising losses listed in [Self-Supervised Learning](https://deepinv.org/user_guide/training/loss.html.md#self-supervised-losses), e.g., [`deepinv.loss.R2RLoss`](https://deepinv.org/api/stubs/deepinv.loss.R2RLoss.html.md#deepinv.loss.R2RLoss) or [`deepinv.loss.SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.html.md#deepinv.loss.SureGaussianLoss) if the noise distribution is known exactly. If the noise parameters are unknown, UNSURE can be used instead, i.e., [`deepinv.loss.SureGaussianLoss`](https://deepinv.org/api/stubs/deepinv.loss.SureGaussianLoss.html.md#deepinv.loss.SureGaussianLoss) with the option `unsure` enabled, and if the noise distribution is unknown altogether, the consistency term can be estimated using the Noise2x family of losses.

At training time, a single splitting is performed for each sample in the batch, however, at evaluation time, the reconstructions are averaged over multiple splittings as specified by `eval_n_samples`.

* **Parameters:**
  * **mask_generator** ([*PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator) *,* *None*) – the generator specifying the distribution of splittings. Defaults to a [`deepinv.physics.generator.BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) with image size specified by `img_size`.
  * **consistency_loss** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss) *,* *None*) – the loss used to compute the consistency term. Defaults to a [`deepinv.loss.MCLoss`](https://deepinv.org/api/stubs/deepinv.loss.MCLoss.html.md#deepinv.loss.MCLoss).
  * **prediction_loss** ([*Loss*](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss) *,* *None*) – the loss used to compute the prediction term. Defaults to a [`deepinv.loss.MCLoss`](https://deepinv.org/api/stubs/deepinv.loss.MCLoss.html.md#deepinv.loss.MCLoss).
  * **transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) *,* *None*) – transformations to be used in training mode for Reynolds averaging (optional).
  * **eval_transform** ([*Transform*](https://deepinv.org/api/stubs/deepinv.transform.Transform.html.md#deepinv.transform.Transform) *,* *None*) – transformations to be used in evaluation mode for Reynolds averaging. It can be used to have true Reynolds averaging at evaluation time and efficient Monte Carlo estimation at training time. If left unspecified, the value of `transform` is used at evaluation time as well.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,*  *...* *]*) – the image size for the fallback splitting scheme (optional). It is only used if `mask_generator` is not specified.

<hr />

* **Example:**

```pycon
>>> import torch
>>> import deepinv as dinv
>>> physics = dinv.physics.Inpainting(img_size=(1, 8, 8), mask=0.5)
>>> model = dinv.models.RAM(pretrained=True)
>>> mask_generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
...     img_size=(1, 8, 8),
...     split_ratio=0.9,
...     pixelwise=True,
... )
>>> train_transform = dinv.transform.Rotate(
...     n_trans=1, multiples=90, positive=True
... ) * dinv.transform.Reflect(n_trans=1, dim=[-1])
>>> eval_transform = dinv.transform.Rotate(
...     n_trans=4, multiples=90, positive=True
... ) * dinv.transform.Reflect(n_trans=2, dim=[-1])
>>> loss = dinv.loss.EquivariantSplittingLoss(
...     mask_generator=mask_generator,
...     consistency_loss=dinv.loss.MCLoss(metric=dinv.metric.MSE()),
...     prediction_loss=dinv.loss.MCLoss(metric=dinv.metric.MSE()),
...     transform=train_transform,
...     eval_transform=eval_transform,
...     eval_n_samples=5,
... )
>>> eq_model = loss.adapt_model(model) # turn into equiv. reconstructor
>>> x = torch.ones((1, 1, 8, 8))
>>> y = physics(x)
>>> x_net = eq_model(y, physics, update_parameters=True)
>>> l = loss(x_net, y, physics, eq_model)
>>> print(l.item() > 0)
True
```

<hr />

* **References:**

* <a id='footcite-sechaud26equivariant'>**[1]**</a> Victor Sechaud, Jérémy Scanvic, Quentin Barthélemy, Patrice Abry, and Julián Tachella. Equivariant Splitting: Self-supervised learning from incomplete data. In *The Fourteenth International Conference on Learning Representations (ICLR)*. 2026.

#### adapt_model(model)

Adapt the reconstructor for equivariant splitting.

It wraps the input reconstructor in a splitting model and optionally in a [`deepinv.models.EquivariantReconstructor`](https://deepinv.org/api/stubs/deepinv.models.EquivariantReconstructor.html.md#deepinv.models.EquivariantReconstructor) if requested.

* **Parameters:**
  **model** ([*Reconstructor*](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)) – the reconstructor to adapt.
* **Returns:**
  the adapted reconstructor.

#### forward(x_net, y, physics, model, \*\*kwargs)

Compute the equivariant splitting loss.

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the reconstructed image.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – the measurement.
  * **physics** ([*Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – the physics operator.
  * **model** ([*Reconstructor*](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)) – the reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the loss value.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-equivariantsplittingloss"></a>

## Examples using `EquivariantSplittingLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
