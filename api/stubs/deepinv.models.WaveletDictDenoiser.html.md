# WaveletDictDenoiser

### *class* deepinv.models.WaveletDictDenoiser(level=3, list_wv=('db8', 'db4'), max_iter=10, non_linearity='soft', mode='zero', wvdim=2, is_complex=False, device='cpu')

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Overcomplete Wavelet denoising with the $\ell_1$ norm.

This denoiser is defined as the solution to the optimization problem:

$$
\underset{x}{\arg\min} \;  \|x-y\|^2 + \lambda \|\Psi x\|_n
$$

where $\Psi$ is an overcomplete wavelet transform, composed of 2 or more wavelets, i.e.,
$\Psi=[\Psi_1,\Psi_2,\dots,\Psi_L]$, $\lambda>0$ is a hyperparameter, and where
$\|\cdot\|_n$ is either the $\ell_1$ norm (`non_linearity="soft"`),
the $\ell_0$ norm (`non_linearity="hard"`) or a variant of the $\ell_0$ norm
(`non_linearity="topk"`) where only the top-k coefficients are kept; see [`deepinv.models.WaveletDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDenoiser.html.md#deepinv.models.WaveletDenoiser) for
more details.

The solution is not available in closed-form, thus the denoiser runs an optimization algorithm for each test image.

#### WARNING
This model requires Pytorch Wavelets (`ptwt`) to be installed. It can be installed with
`pip install ptwt`.

* **Parameters:**
  * **level** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – decomposition level of the wavelet transform.
  * **list_wv** (*Sequence* *[*[*str*](https://docs.python.org/3.9/library/stdtypes.html#str) *]*) – list of mother wavelets. The names of the wavelets can be found in [here](https://wavelets.pybytes.com/). (default: [“db8”, “db4”]).
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) *,* [*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or gpu.
  * **max_iter** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of iterations of the optimization algorithm (default: 10).
  * **non_linearity** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – “soft”, “hard” or “topk” thresholding (default: “soft”)
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – padding mode, “reflect”, “zero”, “constant”, “periodic”, “symmetric” (default: “zero”)
  * **wvdim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension of the wavelet transform (either 2 or 3) (default: 2).
  * **is_complex** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the input is complex-valued (default: False).

#### forward(y, ths=0.1, \*\*kwargs)

Run the model on a noisy image.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image. Assumes a tensor of shape (B, C, H, W) (2D data) or (B, C, D, H, W) (3D data).
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noise level.

#### psi(x, \*\*kwargs)

Returns a flattened list containing the wavelet coefficients for each wavelet.

<a id="sphx-glr-backref-deepinv-models-waveletdictdenoiser"></a>

## Examples using `WaveletDictDenoiser`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example provides a tour of the denoisers in DeepInverse. A denoiser is a model that takes in a noisy image and outputs a denoised version of it. Basically, it solves the following problem:">  <div class="sphx-glr-thumbnail-title">Benchmarking pretrained denoisers</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">  <div class="sphx-glr-thumbnail-title">3D denoising</div>
</div>
<!-- thumbnail-parent-div-close --></div>
