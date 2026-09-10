# MMSE

### *class* deepinv.models.MMSE(dataloader=None, device='cpu', dtype=torch.float32)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Closed-form MMSE denoiser for a Dirac-mixture prior based on a given dataset of images $x_k$.

$$
p(x) = \frac{1}{N} \sum_{k=1}^N \delta(x - x_k)
$$

Given a noisy observation $y = x + \sigma n$ with $n \sim \mathcal{N}(0, I)$, the MMSE estimate is given by:

$$
\mathbb{E}[x | y] = \sum_{k=1}^N x_k w(x_k \vert y)  \quad \text{with} \quad w(x_k \vert y) = \mathrm{softmax}\left( \left(- \frac{1}{\sigma^2}\|y - x_m\|^2 \right)_{m = 1, \cdots, N} \right)_k.
$$

Here, $w(x_k \vert y)$ is the posterior weight of atom $x_k$ knowing the measurement $y$.

* **Parameters:**
  * **dataloader** ([*torch.utils.data.DataLoader*](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Pytorch dataloader or tensor containing the dataset to use as prior.
    If a tensor is provided, it is assumed to contain all the dataset in memory.
    If the dataset is small, using a tensor can significantly speed up computations.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to perform computations on. Default to CPU.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – dtype to compute the estimates. Default to `torch.float32`.
    For large datasets, using `torch.float16` or `torch.bfloat16` can significantly speed up computations.
    In this case, the accumulation is performed in `torch.float32` to avoid numerical issues.

<hr />

* **Examples:**
  ```pycon
  >>> import deepinv as dinv
  >>> import torch
  >>> from torchvision import datasets
  >>> import torchvision.transforms.v2 as v2
  >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  >>> dataset = datasets.MNIST(
  ...        root=".",
  ...        train=False,
  ...        download=True,
  ...        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
  >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
  >>> # Since the MNIST dataset is small, we can also load it entirely in memory as a tensor
  >>> dataloader = torch.cat([data[0] for data in iter(dataloader)]).to(device)
  >>> x = dataloader[0:4]
  >>> denoiser = dinv.models.MMSE(dataloader=dataloader, device=device, dtype=torch.float32)
  >>> sigma = 0.1
  >>> x_noisy = x + sigma * torch.randn_like(x)
  >>> with torch.no_grad():
  ...     x_denoised = denoiser(x_noisy, sigma=sigma)
  ```

#### forward(x, sigma, \*args, verbose=False, \*\*kwargs)

Perform MMSE denoising on input tensor `x` with noise standard deviation `sigma`.

* **Parameters:**
  * **x** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Noisy input tensor of shape `(B, C, H, W)`.
  * **sigma** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *|* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – Noise standard deviation. Can be a float or a tensor of shape `(B,)`.
  * **verbose** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If `True`, display a progress bar during computation. Default is `False`.
* **Returns:**
  Denoised tensor of the same shape as `x`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### to(device=None, dtype=None)

Move the model to a specified device and/or dtype.

* **Parameters:**
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *|* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Device to move the model to.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – Dtype to move the model to.
* **Returns:**
  The model on the specified device and/or dtype.
* **Return type:**
  [MMSE](#deepinv.models.MMSE)

<a id="sphx-glr-backref-deepinv-models-mmse"></a>

## Examples using `MMSE`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
