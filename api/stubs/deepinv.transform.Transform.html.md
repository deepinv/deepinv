# Transform

### *class* deepinv.transform.Transform(\*args, n_trans=1, rng=None, constant_shape=True, flatten_video_input=True, \*\*kwargs)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), [`TimeMixin`](https://deepinv.org/api/stubs/deepinv.utils.TimeMixin.html.md#deepinv.utils.TimeMixin)

Base class for image transforms.

The base transform implements transform arithmetic and other methods to invert transforms and symmetrize functions.

All transforms must implement `_get_params()` to randomly generate e.g. rotation degrees or shift pixels,
and `_transform()` to deterministically transform an image given the params.

To implement a new transform, please reimplement `_get_params()` and `_transform()` (with a `**kwargs` argument).
See respective methods for details.

Also handle deterministic (non-random) transformations by passing in fixed parameter values.

All transforms automatically handle video input (5D of shape `(B,C,T,H,W)`) by flattening the time dimension.

<hr />

* **Examples:**
  Randomly transform an image:
  ```pycon
  >>> import torch
  >>> from deepinv.transform import Shift, Rotate
  >>> from torchvision.transforms import InterpolationMode
  >>> x = torch.rand((1, 1, 2, 2)) # Define random image (B,C,H,W)
  >>> transform = Shift() # Define random shift transform
  >>> transform(x).shape
  torch.Size([1, 1, 2, 2])
  ```

  Deterministically transform an image:
  ```pycon
  >>> y = transform(transform(x, x_shift=[1]), x_shift=[-1])
  >>> torch.all(x == y)
  tensor(True)
  ```

  # Accepts video input of shape (B,C,T,H,W):
  ```pycon
  >>> transform(torch.rand((1, 1, 3, 2, 2))).shape
  torch.Size([1, 1, 3, 2, 2])
  ```

  Multiply transforms to create compound transforms (direct product of groups) - similar to `torchvision.transforms.Compose`:
  ```pycon
  >>> rotoshift = Rotate(
  ...     interpolation_mode=InterpolationMode.BILINEAR
  ... ) * Shift() # Chain rotate and shift transforms
  >>> rotoshift(x).shape
  torch.Size([1, 1, 2, 2])
  ```

  Sum transforms to create stacks of transformed images (along the batch dimension).
  ```pycon
  >>> transform = Rotate(
  ...     interpolation_mode=InterpolationMode.BILINEAR
  ... ) + Shift() # Stack rotate and shift transforms
  >>> transform(x).shape
  torch.Size([2, 1, 2, 2])
  ```

  Randomly select from transforms - similar to `torchvision.transforms.RandomApply`:
  ```pycon
  >>> transform = Rotate(
  ...     interpolation_mode=InterpolationMode.BILINEAR
  ... ) | Shift() # Randomly select rotate or shift transforms
  >>> transform(x).shape
  torch.Size([1, 1, 2, 2])
  ```

  Symmetrize a function by averaging over the group (also known as Reynolds averaging):
  ```pycon
  >>> f = lambda x: x[..., [0]] * x # Function to be symmetrized
  >>> f_s = rotoshift.symmetrize(f)
  >>> f_s(x).shape
  torch.Size([1, 1, 2, 2])
  ```
* **Parameters:**
  * **n_trans** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of transformed versions generated per input image, defaults to 1
  * **rng** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)) – random number generator, if `None`, use [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), defaults to `None`
  * **constant_shape** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, transformed images are assumed to be same shape as input.
    For most transforms, this will not be an issue as automatic cropping/padding should mean all outputs are same shape.
    If False, for certain transforms including [`deepinv.transform.Rotate`](https://deepinv.org/api/stubs/deepinv.transform.Rotate.html.md#deepinv.transform.Rotate),
    `transform` will try to switch off automatic cropping/padding resulting in errors.
    However, `symmetrize` will still work but perform one-by-one (i.e. without collating over batch, which is less efficient).
  * **flatten_video_input** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – accept video (5D) input of shape `(B,C,T,H,W)` by flattening time dim before transforming and unflattening after all operations.

#### \_\_add_\_(other)

Stacks two transforms via the + operation.

* **Parameters:**
  **other** ([*deepinv.transform.Transform*](#deepinv.transform.Transform)) – other transform
* **Returns:**
  (deepinv.transform.Transform) operator which produces stacked transformed images

#### \_\_mul_\_(other)

Chains two transforms via the \* operation.

* **Parameters:**
  **other** ([*deepinv.transform.Transform*](#deepinv.transform.Transform)) – other transform
* **Returns:**
  (deepinv.transform.Transform) chained operator

#### forward(x, \*\*params)

Perform random transformation on image.

Calls `get_params` to generate random params for image, then `transform` to deterministically transform.

For purely deterministic transformation, pass in custom params and `get_params` will be ignored.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape (B,C,H,W)
* **Return torch.Tensor:**
  randomly transformed images concatenated along the first dimension
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### get_params(x)

Randomly generate transform parameters, one set per n_trans.

Params are represented as tensors where the first dimension indexes batch and `n_trans`.
Params store e.g rotation degrees or shift amounts.

Params may be any Tensor-like object. For inverse transforms, params are negated by default.
To change this behavior (e.g. calculate reciprocal for inverse), wrap the param in a `TransformParam` class:
`p = TransformParam(p, neg=lambda x: 1/x)`

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
* **Return dict:**
  keyword args of transform parameters e.g. `{'theta': 30}`
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### identity(x, average=False)

Sanity check function that should do nothing.

This performs forward and inverse transform, which results in the exact original, down to interpolation and padding effects.

Interpolation and padding effects will be visible in non-pixelwise transformations, such as arbitrary rotation, scale or projective transformation.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **average** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – average over `n_trans` transformed versions to get same number as output images as input images. No effect when `n_trans=1`.
* **Return torch.Tensor:**
  $T_g^{-1}T_g x=x$
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### inverse(x, batchwise=True, \*\*params)

Perform random inverse transformation on image (i.e. when not a group).

For purely deterministic transformation, pass in custom params and `get_params` will be ignored.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image
  * **batchwise** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the output dim 0 expands to be of size `len(x) * len(param)` for the params of interest.
    If False, params will attempt to match each image in batch to keep constant `len(out)=len(x)`. No effect when `n_trans==1`
* **Return torch.Tensor:**
  randomly transformed images
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### invert_params(params)

Invert transformation parameters. Pass variable of type `TransformParam` to override negation (e.g. to take reciprocal).

* **Parameters:**
  **params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict)) – transform parameters as dict
* **Return dict:**
  inverted parameters.
* **Return type:**
  [dict](https://docs.python.org/3.9/library/stdtypes.html#dict)

#### symmetrize(f, average=False, collate_batch=True)

Symmetrize a function with a transform and its inverse.

Given a function $f(\cdot):X\rightarrow X$ and a transform $T_g$, returns the group averaged function  $\sum_{i=1}^N T_{g_i}^{-1} f(T_{g_i} \cdot)$ where $N$ is the number of random transformations.

For example, this is useful for Reynolds averaging a function over a group. Set `average=True` to average over `n_trans`.
For example, use `Rotate(n_trans=4, positive=True, multiples=90).symmetrize(f)` to symmetrize f over the entire group.

* **Parameters:**
  * **f** (*Callable* *[* *[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *Any* *]* *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]*) – function acting on tensors.
  * **average** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – monte carlo average over all random transformations (in range `n_trans`) when symmetrising to get same number of output images as input images. No effect when `n_trans=1`.
  * **collate_batch** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, collect `n_trans` transformed images in batch dim and evaluate `f` only once.
    However, this requires `n_trans` extra memory. If `False`, evaluate `f` for each transformation.
    Always will be `False` when transformed images aren’t constant shape.
* **Return Callable[[torch.Tensor, Any], torch.Tensor]:**
  decorated function.
* **Return type:**
  [*Callable*](https://docs.python.org/3.9/library/typing.html#typing.Callable)[[[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [*Any*](https://docs.python.org/3.9/library/typing.html#typing.Any)], [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

#### transform(x, \*\*params)

Transform image given transform parameters.

Given randomly generated params (e.g. rotation degrees), deterministically transform the image x.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image of shape (B,C,H,W)
  * **params** – parameters e.g. degrees or shifts provided as keyword args.
* **Returns:**
  torch.Tensor: transformed image.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-transform-transform"></a>

## Examples using `Transform`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example presents the various datasets, forward physics and models available in DeepInverse for Magnetic Resonance Imaging (MRI) problems:">  <div class="sphx-glr-thumbnail-title">Tour of MRI functionality in DeepInverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates various geometric image transformations implemented in deepinv that can be used in Equivariant Imaging (EI) for self-supervised learning:">  <div class="sphx-glr-thumbnail-title">Image transformations for Equivariant Imaging</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows you how to train a reconstruction network for an MRI inverse problem on a fully self-supervised way, i.e., using measurement data only.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Imaging for MRI.</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Equivariant splitting consists in minimizing a self-supervised loss to train a reconstruction model using measurement data only :footcitesechaud26Equivariant.">  <div class="sphx-glr-thumbnail-title">Self-supervised learning with Equivariant Splitting</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate the use of our deepinv.transform module for use in solving imaging problems. These can be used for:">  <div class="sphx-glr-thumbnail-title">Image transforms for equivariance & augmentations</div>
</div>
<!-- thumbnail-parent-div-close --></div>
