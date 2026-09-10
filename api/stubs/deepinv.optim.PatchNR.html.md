# PatchNR

### *class* deepinv.optim.PatchNR(normalizing_flow=None, pretrained=None, patch_size=6, channels=1, num_layers=5, sub_net_size=256, device='cpu')

Bases: [`Prior`](https://deepinv.org/api/stubs/deepinv.optim.Prior.html.md#deepinv.optim.Prior)

Patch prior via normalizing flows.

The prior is defined as the sum of the negative log-likelihoods of all
(overlapping) patches of the image under a learned normalizing flow model <sup>[1](#footcite-altekruger2023patchnr)</sup>.
Denoting by $P_i x$ the $i$-th patch of image $x$ (out of $N$) and by
$f_\theta$ the normalizing flow with parameters $\theta$, the prior reads

$$
\reg{x} = \frac{1}{N} \sum_{i=1}^{N} -\log p_\theta(P_i x)
$$

where $p_\theta$ is the patch distribution implicitly defined by the flow.
Applying the change-of-variables formula with the standard Gaussian base distribution
$p_z = \mathcal{N}(0, I)$, this expands to

$$
\reg{x} = \frac{1}{N} \sum_{i=1}^{N} \left(
    \frac{1}{2}\| f_\theta(P_i x) \|_2^2
    - \log \left|\det J_{f_\theta}(P_i x)\right|
\right)
$$

where $J_{f_\theta}$ is the Jacobian of $f_\theta$.  Both terms are
computed in a single forward pass through the flow, which returns the latent code
$z = f_\theta(P_i x)$ and the log-determinant
$\log |\det J_{f_\theta}(P_i x)|$ simultaneously.

The forward method evaluates this negative log likelihood.

* **Parameters:**
  * **normalizing_flow** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – describes the normalizing flow of the model. Generally it can be any [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
    supporting backpropagation. It takes a (batched) tensor of flattened patches and the boolean `rev` (default `False`)
    as input and returns `(latent, log_det_jacobian)` as output.
    If `rev=True`, it applies the inverse of the flow.
    When set to `None`, a GLOW-style invertible neural network is built, where the number of
    coupling blocks and the hidden neurons of the sub-networks are determined by `num_layers` and `sub_net_size` respectively.
    If `None`, it is set to [`deepinv.optim.prior.NormalizingFlow`](https://deepinv.org/api/stubs/deepinv.optim.prior.NormalizingFlow.html.md#deepinv.optim.prior.NormalizingFlow) model.
  * **pretrained** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – Define pretrained weights by its path checkpoint, `None` for random initialization,
    `"PatchNR_lodopab_small2"` for the weights from the [limited-angle CT example](https://deepinv.org/auto_examples/optimization/demo_patch_priors_CT.html.md#sphx-glr-auto-examples-optimization-demo-patch-priors-ct-py).
  * **patch_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – size of patches
  * **channels** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of channels for the underlying images/patches.
  * **num_layers** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – defines the number of coupling blocks of the normalizing flow if `normalizing_flow` is `None`.
  * **sub_net_size** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – defines the number of hidden neurons in the subnetworks of the normalizing flow
    if `normalizing_flow` is `None`.
  * **device** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – used device

<hr />

* **Examples:**

```pycon
>>> import torch
>>> import deepinv as dinv
>>> prior = dinv.optim.PatchNR(patch_size=6, channels=1)
>>> x = torch.randn(2, 10, 36)  # (batch, n_patches, patch_size^2 * channels)
>>> nll = prior.fn(x)
>>> nll.shape
torch.Size([2, 10])
```

<hr />

* **References:**

* <a id='footcite-altekruger2023patchnr'>**[1]**</a> Fabian Altekrüger, Alexander Denker, Paul Hagemann, Johannes Hertrich, Peter Maass, and Gabriele Steidl. PatchNR: learning from very few images by patch normalizing flow regularization. *Inverse Problems*, 39(6):064006, 2023.

#### fn(x, \*args, \*\*kwargs)

Evaluates the negative log likelihood function of th PatchNR.

* **Parameters:**
  **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Variable $x$ at which the prior is computed.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) prior $g(x)$.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-optim-patchnr"></a>

## Examples using `PatchNR`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
