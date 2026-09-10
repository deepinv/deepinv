# WaveletPrior

### *class* deepinv.optim.WaveletPrior(level=3, wv='db8', p=1, device='cpu', wvdim=2, is_complex=False, mode='zero', clamp_min=None, clamp_max=None, \*args, \*\*kwargs)

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Wavelet prior $\reg{x} = \|\Psi x\|_{p}$.

$\Psi$ is an orthonormal wavelet transform, and $\|\cdot\|_{p}$ is the $p$-norm, with
$p=0$, $p=1$, or $p=\infty$.

If clamping parameters are provided, the prior writes as $\reg{x} = \|\Psi x\|_{p} + \iota_{[c_{\text{min}}, c_{\text{max}}]}(x)$,
where $\iota_{[c_{\text{min}}, c_{\text{max}}]}(x)$ is the indicator function of the interval $[c_{\text{min}}, c_{\text{max}}]$.

#### NOTE
Following common practice in signal processing, only detail coefficients are regularized, and the approximation
coefficients are left untouched.

#### WARNING
For 3D data, the computational complexity of the wavelet transform cubically with the size of the support. For
large 3D data, it is recommended to use wavelets with small support (e.g. db1 to db4).

* **Parameters:**
  * **level** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – level of the wavelet transform. Default is 3.
  * **wv** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – wavelet name to choose among those available in [pywt](https://pywavelets.readthedocs.io/en/latest/). Default is “db8”.
  * **p** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – $p$-norm of the prior. Default is 1.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – device on which the wavelet transform is computed. Default is “cpu”.
  * **wvdim** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension of the wavelet transform, can be either 2 or 3. Default is 2.
  * **is_complex** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the input is complex-valued. Default is False.
  * **mode** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – padding mode for the wavelet transform (default: “zero”).
  * **clamp_min** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – minimum value for the clamping. Default is None.
  * **clamp_max** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum value for the clamping. Default is None.

#### fn(x, \*args, reduce=True, \*\*kwargs)

Computes the regularizer

$$
\begin{equation}
 {\regname}_{i,j}(x) = \|(\Psi x)_{i,j}\|_{p}
 \end{equation}

$$

where $\Psi$ is an orthonormal wavelet transform, $i$ and $j$ are the indices of the
wavelet sub-bands,  and $\|\cdot\|_{p}$ is the $p$-norm, with
$p=0$, $p=1$, or $p=\infty$. As mentioned in the class description, only detail coefficients
are regularized, and the approximation coefficients are left untouched.

If `reduce` is set to `True`, the regularizer is summed over all detail coefficients, yielding

$$
\regname(x) = \|\Psi x\|_{p}.

$$

If `reduce` is set to `False`, the regularizer is returned as a list of the norms of the detail coefficients.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
  * **reduce** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, the prior is summed over all detail coefficients. Default is True.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $g(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### prox(x, \*args, ths=0.1, gamma=1.0, \*\*kwargs)

Compute the proximity operator of the wavelet prior with the denoiser [`WaveletDenoiser`](https://deepinv.org/api/stubs/deepinv.models.WaveletDenoiser.html.md#deepinv.models.WaveletDenoiser).
Only detail coefficients are thresholded.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the proximity operator is computed.
  * **ths** ([*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float) *,* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – thresholding parameter $\gamma$.
    If `ths` is a tensor, it should be of shape
    `(B,)` (same coefficent for all levels), `(B, n_levels-1)` (one coefficient per level),
    or `(B, n_levels-1, 3)` (one coefficient per subband and per level). `B` should be the same as the batch size of the input or `1`.
    If `non_linearity` equals `"soft"` or `"hard"`, `ths` serves as a (soft or hard)
    thresholding parameter for the wavelet coefficients. If `non_linearity` equals `"topk"`,
    `ths` can indicate the number of wavelet coefficients
    that are kept (if `int`) or the proportion of coefficients that are kept (if `float`).
  * **gamma** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – proximal operator stepsize.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) proximity operator at $x$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### psi(x, \*args, \*\*kwargs)

Applies the (flattening) wavelet decomposition of x.

<a id="sphx-glr-backref-deepinv-optim-waveletprior"></a>

## Examples using `WaveletPrior`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example, we investigate a simple 2D Radio Interferometry (RI) imaging task with deepinverse. The following example and data are taken from :footciteaghabiglou2024r2d2. If you are interested in RI imaging problem and would like to see more examples or try the state-of-the-art algorithms, please check BASPLib.">  <div class="sphx-glr-thumbnail-title">Radio interferometric imaging with deepinverse</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to use a standard wavelet prior for image inpainting. The problem writes as y = Ax + \\epsilon where A is a mask and \\epsilon is the realization of some Gaussian noise. The goal is to recover the original image x from the blurred and noisy image y. The wavelet prior is used to regularize the problem.">  <div class="sphx-glr-thumbnail-title">Image inpainting with wavelet prior</div>
</div><div class="sphx-glr-thumbcontainer" tooltip="This example shows how to implement the LISTA algorithm :footcitegregor2010learning, for a compressed sensing problem. In a nutshell, LISTA is an unfolded proximal gradient algorithm involving a soft-thresholding proximal operator with learnable thresholding parameters.">  <div class="sphx-glr-thumbnail-title">Learned Iterative Soft-Thresholding Algorithm (LISTA) for compressed sensing</div>
</div>
<!-- thumbnail-parent-div-close --></div>
