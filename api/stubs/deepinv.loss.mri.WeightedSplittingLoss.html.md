# WeightedSplittingLoss

### *class* deepinv.loss.mri.WeightedSplittingLoss(mask_generator, physics_generator, metric=None)

Bases: [`SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss)

K-Weighted Splitting Loss

Implements the K-weighted Noisier2Noise-SSDU loss from Millard and Chiew<sup>[1](#footcite-millard2023theoretical)</sup>.
The loss is designed for problems where measurements are observed as $y_i=M_iAx$,
where $M_i$ is a random mask, such as in [`MRI`](https://deepinv.org/api/stubs/deepinv.physics.MRI.html.md#deepinv.physics.MRI) where `A` is the Fourier transform.
The loss is defined as follows, using notation from [`deepinv.loss.SplittingLoss`](https://deepinv.org/api/stubs/deepinv.loss.SplittingLoss.html.md#deepinv.loss.SplittingLoss):

$$
\frac{m}{m_2}\| (1-\mathbf{K})^{-1/2} (y_2 - A_2 \inversef{y_1}{A_1})\|^2
$$

where $\mathbf{K}$ is derived from the probability density function (pdf) of the (original) acceleration mask and (further) splitting mask:

$$
\mathbf{K}=(\mathbb{I}_n-\tilde{\mathbf{P}}\mathbf{P})^{-1}(\mathbb{I}_n-\mathbf{P})
$$

and $\mathbf{P}=\mathbb{E}[\mathbf{M}_i],\tilde{\mathbf{P}}=\mathbb{E}[\mathbf{M}_1]$ i.e. the average imaging mask and splitting mask, respectively.
At inference, the original whole measurement $y$ is used as input.

#### NOTE
To match the original paper, the loss should be used with the splitting mask [`deepinv.physics.generator.MultiplicativeSplittingMaskGenerator`](https://deepinv.org/api/stubs/deepinv.physics.generator.MultiplicativeSplittingMaskGenerator.html.md#deepinv.physics.generator.MultiplicativeSplittingMaskGenerator)
where the input additional subsampling mask should be the same type as that used to generate the measurements.

Note the method was originally proposed for accelerated MRI problems (where the measurements are generated via a mask generator).

Note also that we assume that all masks are 1D mask in the image width dimension repeated in all other dimensions.

If the input data varies in shape, the loss will dynamically recalculate the weight. However, this will be slower every time the weight must be recalculated.

* **Parameters:**
  * **mask_generator** ([*deepinv.physics.generator.BernoulliSplittingMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator)) – splitting mask generator for further subsampling.
  * **physics_generator** ([*deepinv.physics.generator.BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)) – original mask generator used to generate the measurements.
  * **metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – metric used for computing data consistency, which is set as the mean squared error by default.

<hr />

* **Example:**

```pycon
>>> import torch
>>> from deepinv.physics.generator import GaussianMaskGenerator, MultiplicativeSplittingMaskGenerator
>>> from deepinv.loss.mri import WeightedSplittingLoss
>>> physics_generator = GaussianMaskGenerator((128, 128), acceleration=4)
>>> split_generator = GaussianMaskGenerator((128, 128), acceleration=2)
>>> mask_generator = MultiplicativeSplittingMaskGenerator((1, 128, 128), split_generator)
>>> loss = WeightedSplittingLoss(mask_generator, physics_generator)
```

<hr />

* **References:**

* <a id='footcite-millard2023theoretical'>**[1]**</a> Charles Millard and Mark Chiew. A theoretical framework for self-supervised mr image reconstruction using sub-sampling via variable density noisier2noise. *IEEE transactions on computational imaging*, 9:707–720, 2023.

#### *class* WeightedMetric(mask_generator, physics_generator, pixel_metric)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Wraps metric to apply weight on inputs

Note `mask_generator` and `physics_generator` are only used to regenerate the weight in the case that y has different shapes during training.

* **Parameters:**
  * **torch.Tensor** – loss weight.
  * **mask_generator** ([*deepinv.physics.generator.BernoulliSplittingMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator)) – splitting mask generator for further subsampling.
  * **physics_generator** ([*deepinv.physics.generator.BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)) – original mask generator used to generate the measurements.
  * **pixel_metric** ([*Metric*](https://deepinv.org/api/stubs/deepinv.loss.metric.Metric.html.md#deepinv.loss.metric.Metric) *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – loss metric.
  * **expand** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether expand weight to input dims

#### forward(y1, y2)

Weighted metric forward pass.

#### *static* compute_weight(mask_generator, physics_generator, eps=1e-9, img_size=None)

Compute weight for K-weighted splitting loss where K is a diagonal matrix of shape `(H, W)`,
and returned weight is a tensor of shape `(1, W)`.

Estimates the 1D PDFs of the mask generators empirically.

* **Parameters:**
  * **mask_generator** ([*deepinv.physics.generator.BernoulliSplittingMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BernoulliSplittingMaskGenerator.html.md#deepinv.physics.generator.BernoulliSplittingMaskGenerator)) – splitting mask generator for further subsampling.
  * **physics_generator** ([*deepinv.physics.generator.BaseMaskGenerator*](https://deepinv.org/api/stubs/deepinv.physics.generator.BaseMaskGenerator.html.md#deepinv.physics.generator.BaseMaskGenerator)) – original mask generator used to generate the measurements.
  * **eps** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – small value to avoid division by zero.
  * **img_size** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – desired mask shape `(H, W)`. If `None`, use default provided in `physics_generator` and `mask_generator`.

#### *property* name *: [bool](https://docs.python.org/3.9/library/functions.html#bool)*

The name of the loss function. This attribute is deprecated in favor of the class name and it will be removed in a future version.

<a id="sphx-glr-backref-deepinv-loss-mri-weightedsplittingloss"></a>

## Examples using `WeightedSplittingLoss`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate scan-specific self-supervised learning, that is, learning to reconstruct MRI scans from a single accelerated sample without ground truth.">  <div class="sphx-glr-thumbnail-title">Scan-specific zero-shot SSDU for MRI</div>
</div>
<!-- thumbnail-parent-div-close --></div>
