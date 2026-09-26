# BilateralFilter

### *class* deepinv.models.BilateralFilter(device='cpu')

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Bilateral filter.

The bilateral filter as it was introduced in Tomasi and Manduchi<sup>[1](#footcite-tomasi1998bilateral)</sup>
where each output pixel is a normalized weighted average of neighboring pixels in a spatial window.
The weights factor into a spatial kernel and a range kernel:

$$
\hat{x}_i = (1 / W_i) * \sum_{j \in \omega(i)} k_d(i - j) k_r(x_i - x_j) x_j
$$

with Gaussian spatial kernel $k_d$ and Gaussian range kernel $k_d$.
The spatial standard deviation $\sigma_d$ controls how fast the kernel
decays with distance, while the range standard deviation $\sigma_r$
controls how strongly intensity differences are penalized.

<hr />

* **Example:**

```pycon
>>> import torch
>>> import deepinv as dinv
>>> x = dinv.utils.load_example("butterfly.png")
>>> rng = torch.Generator().manual_seed(0)
>>> physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(rng=rng))
>>> y = physics(x)
>>> model = dinv.models.BilateralFilter()
>>> x_hat = model(y,sigma_d=1.1,sigma_r=0.3,window_size=9)
>>> dinv.metric.PSNR()(x_hat, x) > 26
tensor([True])
```

<hr />

* **References:**

* <a id='footcite-tomasi1998bilateral'>**[1]**</a> C. Tomasi and R. Manduchi. Bilateral filtering for gray and color images. In *Sixth International Conference on Computer Vision (IEEE Cat. No.98CH36271)*, 839–846. 1998.

#### forward(x, sigma=None, , sigma_d=1, sigma_r=1, window_size=5, \*\*kwargs)

Apply a bilateral filter to the input x.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Input tensor of shape (B, C, H, W) or (C, H, W).
  * **sigma_d** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Spatial standard deviation (spatial domain). Larger values yield larger effective receptive fields.
  * **sigma_r** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – Range standard deviation (intensity domain). Larger values yield stronger smoothing across edges.
  * **window_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Size of the (square) spatial window. Must be an odd integer.
* **Returns:**
  Filtered image.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
