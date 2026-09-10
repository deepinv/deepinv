# SplittingLoss

### *class* deepinv.loss.SplittingLoss(metric=None, split_ratio=0.9, mask_generator=None, eval_n_samples=5, eval_split_input=True, eval_split_output=False, pixelwise=True, normalize_loss=True)

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Measurement splitting loss.

Implements measurement splitting loss. Splits the measurement and forward operator $A$ (of size $m$)
into two smaller pairs  $(y_1,A_1)$ (of size $m_1$) and  $(y_2,A_2)$ (of size $m_2$) ,
to compute the self-supervised loss:

$$
\frac{m}{m_2}\| y_2 - A_2 \inversef{y_1}{A_1}\|^2
$$

where $R$ is the trainable network, $A_1 = M_1 A, A_2 = M_2 A$, and $M_i$ are randomly
generated masks (i.e. diagonal matrices) such that $M_1+M_2=\mathbb{I}_m$.

See [Self-supervised learning with measurement splitting](https://deepinv.org/auto_examples/self-supervised-learning/demo_splitting_loss.html.md#sphx-glr-auto-examples-self-supervised-learning-demo-splitting-loss-py) for usage example.

#### NOTE
If the forward operator has its own subsampling mask $M_{A}$, e.g. [`deepinv.physics.Inpainting`](https://deepinv.org/api/stubs/deepinv.physics.Inpainting.html.md#deepinv.physics.Inpainting)
or [`deepinv.physics.MRI`](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI),
the splitting masks will be subsets of the physics’ mask such that $M_1+M_2=M_{A}$

This loss was used for MRI in SSDU Yaman *et al.*<sup>[1](#footcite-yaman2020self)</sup> for MRI, Noise2Inverse Hendriksen *et al.*<sup>[2](#footcite-hendriksen2020noise2inverse)</sup> for CT, as well as numerous other papers.
Note we implement the multi-mask strategy proposed by Yaman *et al.*<sup>[1](#footcite-yaman2020self)</sup>.

By default, the error is computed using the MSE metric, however any appropriate metric can be used.

#### WARNING
The model should be adapted before training using the method [`adapt_model`](#deepinv.loss.SplittingLoss.adapt_model)
to include the splitting mechanism at the input.

#### NOTE
To obtain the best test performance, the trained model should be averaged at test time
over multiple realizations of the splitting, i.e.
$\hat{x} = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}$. To disable this, set `eval_n_samples=1`.

#### NOTE
To disable measurement splitting (and use the full input) at evaluation time, set `eval_split_input=False`. This is done in SSDU Yaman *et al.*<sup>[1](#footcite-yaman2020self)</sup>.

#### NOTE
This loss allows training over images of varying shapes.

#### SEE ALSO
[`deepinv.loss.mri.Artifact2ArtifactLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Artifact2ArtifactLoss.html.md#deepinv.loss.mri.Artifact2ArtifactLoss), [`deepinv.loss.mri.Phase2PhaseLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.Phase2PhaseLoss.html.md#deepinv.loss.mri.Phase2PhaseLoss), [`deepinv.loss.mri.WeightedSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.WeightedSplittingLoss.html.md#deepinv.loss.mri.WeightedSplittingLoss), [`deepinv.loss.mri.RobustSplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.mri.RobustSplittingLoss.html.md#deepinv.loss.mri.RobustSplittingLoss)
: Specialized splitting losses and their extensions for MRI applications.

* **Parameters:**
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency, which is set as the mean squared error by default.
  * **split_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – splitting ratio, should be between 0 and 1. The size of $y_1$ increases
    with the splitting ratio. Ignored if `mask_generator` passed.
  * **mask_generator** ([*deepinv.physics.generator.BernoulliSplittingMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) *,* *None*) – function to generate the mask. If
    None, the [`deepinv.physics.generator.BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) is used, with the parameters `split_ratio` and `pixelwise`.
  * **eval_n_samples** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of samples used for averaging at evaluation time. Must be greater than 0.
  * **eval_split_input** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, perform input measurement splitting during evaluation. If False, use full measurement at eval (no MC samples are performed and eval_split_output will have no effect)
  * **eval_split_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – at evaluation time, pass the output through the output mask too.
    i.e. $(\sum_{j=1}^N M_2^{(j)})^{-1} \sum_{i=1}^N M_2^{(i)} \inversef{y_1^{(i)}}{A_1^{(i)}}$.
    Only valid when $y$ is same domain (and dimension) as $x$. Although better results may be observed on small datasets, more samples must be used for bigger images. Defaults to `False`.
  * **pixelwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, create pixelwise splitting masks i.e. zero all channels simultaneously. Ignored if `mask_generator` passed.
  * **normalize_loss** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to normalize loss by the target size

<hr />

* **Example:**

```pycon
>>> import torch
>>> import deepinv as dinv
>>> physics = dinv.physics.Inpainting(img_size=(1, 8, 8), mask=0.5)
>>> model = dinv.models.MedianFilter()
>>> loss = dinv.loss.SplittingLoss(split_ratio=0.9, eval_n_samples=2)
>>> model = loss.adapt_model(model) # important step!
>>> x = torch.ones((1, 1, 8, 8))
>>> y = physics(x)
>>> x_net = model(y, physics, update_parameters=True) # save random mask in forward pass
>>> l = loss(x_net, y, physics, model)
>>> print(l.item() > 0)
True
```

<hr />

* **References:**

* <a id='footcite-yaman2020self'>**[1]**</a> Burhaneddin Yaman, Seyed Amir Hossein Hosseini, Steen Moeller, Jutta Ellermann, Kâmil Uğurbil, and Mehmet Akçakaya. Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data. *Magnetic resonance in medicine*, 84(6):3172–3191, 2020.
* <a id='footcite-hendriksen2020noise2inverse'>**[2]**</a> Allard Adriaan Hendriksen, Daniël Maria Pelt, and K Joost Batenburg. Noise2inverse: self-supervised deep convolutional denoising for tomography. *IEEE Transactions on Computational Imaging*, 6:1320–1335, 2020.

#### *class* SplittingModel(model, split_ratio, mask_generator, eval_n_samples, eval_split_input, eval_split_output, pixelwise)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Model wrapper when using SplittingLoss.

Performs input splitting during forward pass. At evaluation,
perform forward passes for multiple realisations of splitting mask and average.

* **Parameters:**
  * **model** ([*deepinv.models.Reconstructor*](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)) – base model
  * **split_ratio** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – splitting ratio, should be between 0 and 1. The size of $y_1$ increases
    with the splitting ratio. Ignored if `mask_generator` passed.
  * **mask_generator** ([*deepinv.physics.generator.PhysicsGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.PhysicsGenerator.html.md#deepinv.physics.generator.PhysicsGenerator) *,* *None*) – function to generate the mask. If
    None, the [`deepinv.physics.generator.BernoulliSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator) is used, with the parameters `split_ratio` and `pixelwise`.
  * **eval_n_samples** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Number of samples used for averaging at evaluation time. Must be greater than 0.
  * **eval_split_input** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, perform input measurement splitting during evaluation. If False, use full measurement at eval (no MC samples are performed and eval_split_output will have no effect)
  * **eval_split_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – at evaluation time, pass the output through the output mask too.
    i.e. $(\sum_{j=1}^N M_2^{(j)})^{-1} \sum_{i=1}^N M_2^{(i)} \inversef{y_1^{(i)}}{A_1^{(i)}}$.
    Only valid when $y$ is same domain (and dimension) as $x$. Although better results may be observed on small datasets, more samples must be used for bigger images. Defaults to `False`.
  * **pixelwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, create pixelwise splitting masks i.e. zero all channels simultaneously. Ignored if `mask_generator` passed.

#### forward(y, physics, update_parameters=False)

Adapted model forward pass for input splitting. During training, only one splitting realisation is performed for computational efficiency.

#### *static* split(mask, y, physics=None)

Perform splitting given mask

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – splitting mask
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input data
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics to split, retaining its original noise model. If `None`, only $y$ is split.

#### adapt_model(model)

Apply random splitting to input.

This method modifies a reconstruction
model $R$ to include the splitting mechanism at the input:

$$
\hat{R}(y, A) = \frac{1}{N}\sum_{i=1}^N \inversef{y_1^{(i)}}{A_1^{(i)}}
$$

where $N\geq 1$ is the number of Monte Carlo samples,
and $y_1^{(i)}$ and $A_1^{(i)}$ are obtained by
randomly splitting the measurements $y$ and operator $A$.
During training (i.e. when `model.train()`), we use only one sample, i.e. $N=1$
for computational efficiency, whereas at test time, we use multiple samples for better performance.
For other parameters that control how splitting is applied, see the class parameters.

* **Parameters:**
  **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction model.
* **Returns:**
  ([`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) Model modified for evaluation.
* **Return type:**
  [SplittingModel](#deepinv.loss.SplittingLoss.SplittingModel)

#### forward(x_net, y, physics, model, \*\*kwargs)

Computes the measurement splitting loss

* **Parameters:**
  * **x_net** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – reconstructions.
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction function.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) loss.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

#### *static* split(mask, y, physics=None)

Perform splitting given mask

* **Parameters:**
  * **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – splitting mask of shape (B,C,H,W)
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input data of shape (B,C,…,H,W)
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – physics to split, retaining its original noise model. If `None`, only $y$ is split.

<a id="sphx-glr-backref-deepinv-loss-splittingloss"></a>

## Examples using `SplittingLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the self-supervised Artifact2Artifact loss for solving an undersampled sequential MRI reconstruction problem without ground truth.">  <div class="sphx-glr-thumbnail-title">Self-supervised MRI reconstruction with Artifact2Artifact</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised learning with measurement splitting, to train a denoiser network on the MNIST dataset. The physics here is noisy computed tomography, as is the case in Noise2Inverse :footcitehendriksen2020noise2inverse. Note this example can also be easily applied to undersampled multicoil MRI as is the case in SSDU :footciteyaman2020self.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with measurement splitting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
