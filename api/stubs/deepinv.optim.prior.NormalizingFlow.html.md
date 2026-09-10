# NormalizingFlow

### *class* deepinv.optim.prior.NormalizingFlow(dimension, num_layers, subnet, clamp=1.6)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Sequential normalizing flow built from GLOW-style affine coupling blocks.

The flow maps an input sample `x` to a latent representation `z` by passing
it through `num_layers` invertible coupling blocks in sequence.  The
log-determinant of the full Jacobian is accumulated additively across blocks.
Setting `rev=True` runs the blocks in reverse order to recover the original
sample from a latent code.

The architecture follows the generative flow of Kingma and Dhariwal<sup>[1](#footcite-kingma2018glow)</sup>,
using [`deepinv.optim.prior.GLOWCouplingBlock`](https://deepinv.org/api/stubs/deepinv.optim.prior.GLOWCouplingBlock.html.md#deepinv.optim.prior.GLOWCouplingBlock) as the building block.

* **Parameters:**
  * **dimension** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – dimension of each input sample (flattened patch size).
  * **num_layers** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of coupling blocks to stack.
  * **subnet** (*Callable*) – a callable `subnet(channels_in, channels_out) -> nn.Module`
    that constructs the subnetworks used inside each coupling block.
  * **clamp** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – soft-clamping magnitude passed to every coupling block. Default is `1.6`.

<hr />

* **Examples:**

```pycon
>>> import torch
>>> import torch.nn as nn
>>> subnet = lambda c_in, c_out: nn.Sequential(
...     nn.Linear(c_in, 32), nn.ReLU(),
...     nn.Linear(32, 32), nn.ReLU(),
...     nn.Linear(32, c_out),
... )
>>> flow = NormalizingFlow(dimension=8, num_layers=2, subnet=subnet)
>>> x = torch.randn(4, 8)
>>> z, log_det = flow(x)
>>> z.shape
torch.Size([4, 8])
>>> log_det.shape
torch.Size([4])
>>> x_rec, _ = flow(z, rev=True)  # inverse flow recovers the input
>>> torch.allclose(x, x_rec, atol=1e-5)
True
```

<hr />

* **References:**

* <a id='footcite-kingma2018glow'>**[1]**</a> Durk P Kingma and Prafulla Dhariwal. Glow: generative flow with invertible 1x1 convolutions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL: [https://proceedings.neurips.cc/paper_files/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf).

#### forward(x, rev=False)

Passes the input through all coupling blocks sequentially.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `(N, dimension)`.
  * **rev** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if `True`, applies the blocks in reverse order (inverse flow). Default is `False`.
* **Returns:**
  tuple `(z, log_det)` where `z` ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) is the latent
  representation of shape `(N, dimension)` and `log_det` ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor))
  is the total log-determinant of the Jacobian of shape `(N,)`.

<a id="sphx-glr-backref-deepinv-optim-prior-normalizingflow"></a>

## Examples using `NormalizingFlow`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="In this example we use patch priors for limited angle computed tomography. More precisely, we consider the inverse problem y = \\mathrm{noisy}(Ax), where A is the discretized Radon transform with 100 equispace angles between 20 and 160 degrees. For the reconstruction, we minimize the variational problem">  <div class="sphx-glr-thumbnail-title">Patch priors for limited-angle computed tomography</div>
</div>
<!-- thumbnail-parent-div-close --></div>
