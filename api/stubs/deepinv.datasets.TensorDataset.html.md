# TensorDataset

### *class* deepinv.datasets.TensorDataset(, x=None, y=None, params=None, use_dict_output=False)

Bases: [`ImageDataset`](https://deepinv.org/api/stubs/deepinv.datasets.ImageDataset.html.md#deepinv.datasets.ImageDataset)

Dataset wrapping data explicitly passed as tensors.

This dataset can be used to return ground truth `x`, ground truth and measurements `(x, y)`, or measurements only `(y)`.
All input tensors must be of shape `(N, ...)` and of same `N` where N is the number of samples and … represents the data dimensions.

#### TIP
Alternatively, you can use `use_dict_output=True` to return a dict with at least keys `"x"` or `"y"`, and `"params"` instead of a tuple. This is recommended for better readability and flexibility in returned outputs.

Optionally, `params` are returned too.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional input ground truth tensor `x`
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* *None*) – optional input measurement tensor `y`
  * **params** ([*dict*](https://docs.python.org/3.9/library/stdtypes.html#dict) *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *]* *,* *None*) – optional input physics parameters `params` of format `{"str": Tensor}`
  * **use_dict_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to return output as dict with keys `"x"`, `"y"`, `"params"``  instead of tuple. Defaults to `False` for backward compatibility.

<hr />

Examples:

Construct a dataset from a single measurement only:

```pycon
>>> import torch
>>> from deepinv.datasets import TensorDataset
>>> y = torch.rand(1, 3, 8, 8) # B,C,H,W
>>> dataset = TensorDataset(y=y)
>>> x, y = dataset[0]
>>> x
nan
>>> y.shape
torch.Size([3, 8, 8])
```

Construct a dataset from a ground truth batch:

```pycon
>>> x = torch.rand(4, 3, 8, 8)  # 4 samples of 3-channel 8x8 images
>>> dataset = TensorDataset(x=x)
>>> dataset[0].shape
torch.Size([3, 8, 8])
```

<a id="sphx-glr-backref-deepinv-datasets-tensordataset"></a>

## Examples using `TensorDataset`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use DeepInverse with your own dataset.">  <div class="sphx-glr-thumbnail-title">Bring your own dataset</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to perform inference on and fine-tune the Reconstruct Anything Model (RAM) foundation model :footciteterris2025reconstruct to solve inverse problems.">  <div class="sphx-glr-thumbnail-title">Inference and fine-tune a foundation model</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.">  <div class="sphx-glr-thumbnail-title">Low-field MRI denoising without ground truth</div>
</div>
<!-- thumbnail-parent-div-close --></div>
