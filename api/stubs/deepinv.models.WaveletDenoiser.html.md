# WaveletDenoiser

### *class* deepinv.models.WaveletDenoiser(level=3, wv='db8', device='cpu', non_linearity='soft', mode='zero', wvdim=2, is_complex=False)

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Orthogonal Wavelet denoising with the $\ell_1$ norm.

This denoiser is defined as the solution to the optimization problem:

$$
\underset{x}{\arg\min} \;  \|x-y\|^2 + \gamma \|\Psi x\|_n
$$

where $\Psi$ is an orthonormal wavelet transform, $\lambda>0$ is a hyperparameter, and where
$\|\cdot\|_n$ is either the $\ell_1$ norm (`non_linearity="soft"`) or
the $\ell_0$ norm (`non_linearity="hard"`). A variant of the $\ell_0$ norm is also available
(`non_linearity="topk"`), where the thresholding is done by keeping the $k$ largest coefficients
in each wavelet subband and setting the others to zero.

The solution is available in closed-form, thus the denoiser is cheap to compute.

#### WARNING
This model requires Pytorch Wavelets (`ptwt`) to be installed. It can be installed with
`pip install ptwt`.

* **Parameters:**
  * **level** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – decomposition level of the wavelet transform
  * **wv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – mother wavelet (follows the [PyWavelets convention](https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html)) (default: “db8”)
  * **non_linearity** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – `"soft"`, `"hard"` or `"topk"` thresholding (default: `"soft"`).
    If `"topk"`, only the top-k wavelet coefficients are kept.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – padding mode for the wavelet transform (default: “zero”).
  * **wvdim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension of the wavelet transform (either 2 or 3) (default: 2).
  * **is_complex** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the input is complex-valued (default: False).
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – cpu or gpu

#### crop_output(x, padding)

Crop the output to make it compatible with the wavelet transform.

#### dwt(x)

Applies the wavelet decomposition.

#### flatten_coeffs(dec)

Flattens the wavelet coefficients and returns them in a single torch vector of shape (n_coeffs,).

#### forward(x, ths=0.1, \*\*kwargs)

Run the model on a noisy image.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy image. Assumes a tensor of shape (B, C, H, W) (2D data) or (B, C, D, H, W) (3D data).
  * **ths** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – thresholding parameter $\gamma$.
    If `ths` is a tensor, it should be of shape
    `(B,)` (same coefficient for all levels), `(B, n_levels-1)` (one coefficient per level),
    or `(B, n_levels-1, 3)` (one coefficient per subband and per level). `B` should be the same as the batch size of the input or `1`.
    If `non_linearity` equals `"soft"` or `"hard"`, `ths` serves as a (soft or hard)
    thresholding parameter for the wavelet coefficients. If `non_linearity` equals `"topk"`,
    `ths` can indicate the number of wavelet coefficients
    that are kept (if `int`) or the proportion of coefficients that are kept (if `float`).

#### hard_threshold_topk(x, ths=0.1)

Hard thresholding of the wavelet coefficients by keeping only the top-k coefficients and setting the others to
0.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – wavelet coefficients.
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int)) – top k coefficients to keep. If `float`, it is interpreted as a proportion of the total
    number of coefficients. If `int`, it is interpreted as the number of coefficients to keep.

#### iwt(coeffs)

Applies the wavelet recomposition.

#### pad_input(x)

Pad the input to make it compatible with the wavelet transform.

#### prox_l0(x, ths=0.1)

Hard thresholding of the wavelet coefficients.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – wavelet coefficients of shape (B, C, H, W) or (B, C, D, H, W).
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – threshold of shape (B,) or scalar. If scalar, same threshold is used for all elements in batch.

#### prox_l1(x, ths=0.1)

Soft thresholding of the wavelet coefficients.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – wavelet coefficients.
  * **ths** ([*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – threshold.

#### *static* psi(x, wavelet='db2', level=2, dimension=2, mode='zero')

Returns a flattened list containing the wavelet coefficients.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input image.
  * **wavelet** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – mother wavelet.
  * **level** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – decomposition level.
  * **dimension** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension of the wavelet transform (either 2 or 3).
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – padding mode for the wavelet transform (default: “zero”).

#### reshape_ths(ths, level)

Reshape the thresholding parameter in the appropriate format, i.e. either:
: - a list of 3 elements, or
  - a tensor of 3 elements.

Since the approximation coefficients are not thresholded, we do not need to provide a thresholding parameter,
ths has shape (n_levels-1, 3).

#### threshold_2D(coeffs, ths)

Thresholds coefficients of the 2D wavelet transform.

#### threshold_3D(coeffs, ths)

Thresholds coefficients of the 3D wavelet transform.

#### threshold_ND(coeffs, ths)

Apply thresholding to the wavelet coefficients of arbitrary dimension.

#### threshold_func(x, ths)

Apply thresholding to the wavelet coefficients.

<a id="sphx-glr-backref-deepinv-models-waveletdenoiser"></a>

## Examples using `WaveletDenoiser`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use variational 3D denoisers for denoising a 3D image. We first apply a standard soft-thresholding wavelet denoiser to a 3D brain MRI volume, as well as a 3D TV denoiser. We then extend the wavelet denoiser objective to a redundant dictionary of wavelet bases, which does not admit a closed-form solution. We solve the denoising problem using the Dykstra-like algorithm.">  <div class="sphx-glr-thumbnail-title">3D denoising</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="Image inpainting consists in solving y = Ax where A is a mask operator. This problem can be reformulated as the following minimization problem:">  <div class="sphx-glr-thumbnail-title">Unfolded Chambolle-Pock for constrained image inpainting</div>
</div>
<!-- thumbnail-parent-div-close --></div>
