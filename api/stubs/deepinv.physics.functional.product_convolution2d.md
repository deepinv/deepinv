# product_convolution2d

### deepinv.physics.functional.product_convolution2d(x, w, h, padding='valid', use_fft=False, mask_first=True)

Product-convolution operator in 2d. Details available in the paper Escande and Weiss<sup>[1](#footcite-escande2017approximation)</sup>.

If `mask_first=True` (default), this forward operator performs

$$
y = \sum_{k=1}^K h_k \star (w_k \odot x)
$$

whereas if `mask_first=False`, the multipliers are applied after the convolutions, i.e.

$$
y = \sum_{k=1}^K w_k \odot (h_k \star x)
$$

where $\star$ is a convolution, $\odot$ is a Hadamard product, $w_k$ are multipliers $h_k$ are filters.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor of size $(B, C, H, W)$
  * **w** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor of size $(b, c, K, H, W)$. $b \in \{1, B\}$ and $c \in \{1, C\}$.
    If `mask_first=False` and `padding='valid'`, the spatial size of the multipliers should match the size of
    the convolution output instead, i.e. $(b, c, K, H-h+1, W-w+1)$.
  * **h** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Tensor of size $(b, c, K, h, w)$. $b \in \{1, B\}$ and $c \in \{1, C\}$, $h\leq H$ and $w\leq W$.
  * **padding** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – ( options = `'valid'`, `'circular'`, `'replicate'`, `'reflect'` or `'constant'`). If `padding = `'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.
  * **use_fft** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to use FFT-based convolutions. If `True`, it uses FFT-based convolutions which can be faster for large kernels.
  * **mask_first** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the multipliers are applied before (`True`, default) or after (`False`) the convolutions.
* **Returns:**
  [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) the blurry image.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<hr />

* **References:**

* <a id='footcite-escande2017approximation'>**[1]**</a> Paul Escande and Pierre Weiss. Approximation of integral operators using product-convolution expansions. *Journal of Mathematical Imaging and Vision*, 58:333–348, 2017.
