# liu_jia_pad

### deepinv.physics.functional.liu_jia_pad(x, , padding)

Liu-Jia Padding

Real-world blurry images have decorrelated opposite boundaries unlike images synthetically blurred using circular filters. This make the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding <sup>[1](#footcite-liu2008reducing)</sup> is a pre-processing step that pads the input image to make it have smooth circular boundaries while preserving the original spectral content as much as possible.

The implementation is adapted from [the one](https://github.com/cszn/USRNet) featured in the work of Zhang *et al.*<sup>[2](#footcite-zhang2020deep)</sup>.

The padded tensor has shape $(B, C, H + 2 * \text{pad}_h, W + 2 * \text{pad}_w)$ where $\text{pad}_h$ and $\text{pad}_w$ are the vertical and horizontal padding respectively.

#### NOTE
Padding a single direction is not supported and a [`ValueError`](https://docs.python.org/3.9/library/exceptions.html#ValueError)
will be raised if only one of the two padding values is non-zero.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input image of shape (B, C, H, W)
  * **padding** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple) *(*[*int*](https://docs.python.org/3.9/library/functions.html#int) *,* [*int*](https://docs.python.org/3.9/library/functions.html#int) *)*) – Left/right padding, and top/bottom padding (px).
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Padded image
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<hr />

* **References:**

* <a id='footcite-liu2008reducing'>**[1]**</a> Renting Liu and Jiaya Jia. Reducing boundary artifacts in image deconvolution. In *2008 15th IEEE International Conference on Image Processing*, 505–508. IEEE, 2008.
* <a id='footcite-zhang2020deep'>**[2]**</a> Kai Zhang, Luc Van Gool, and Radu Timofte. Deep unfolding network for image super-resolution. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 3217–3226. 2020.

## Examples using `liu_jia_pad`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="Real-world blurry images have decorrelated opposite boundaries, unlike images synthetically blurred using circular filters. This makes the use of spectral deconvolution methods (inverse filtering, Wiener filtering) impractical and prone to ringing artifacts. Liu-Jia padding &lt;deepinv.physics.functional.liu_jia_pad&gt; :footciteliu2008reducing is a pre-processing step that pads the input image to make it have smooth circular boundaries, while preserving the original spectral content as much as possible.">  <div class="sphx-glr-thumbnail-title">Spectral Methods for Non-Circular Deblurring with Liu-Jia Padding</div>
</div>
<!-- thumbnail-parent-div-close --></div>
