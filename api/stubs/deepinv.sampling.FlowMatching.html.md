# FlowMatching

### *class* deepinv.sampling.FlowMatching(a_t=lambda t: ..., a_prime_t=lambda t: ..., b_t=lambda t: ..., b_prime_t=lambda t: ..., T=0.99, alpha=0.0, denoiser=None, solver=None, dtype=torch.float64, device=torch.device('cpu'), \*args, \*\*kwargs)

Bases: [`EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE)

Generative Flow Matching process.

It corresponds to the reverse-time SDE of the following forward-time noising process, which corresponds to a linear interpolation between data and Gaussian noise:

$$
x_t = a_t x_0 + b_t z \quad \mbox{ where } x_0 \sim p_{data}  \mbox{ and } z \sim \mathcal{N}(0, I)

$$

The schedulers $a(t)$ and $b(t)$ must satisfy $a(0) = 1$, $b(0) = 0$, $a(1) = 0$, and $b(1) = 1$.

Compared to the EDM formulation in [`deepinv.sampling.EDMDiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.EDMDiffusionSDE.html.md#deepinv.sampling.EDMDiffusionSDE), the scale $s(t)$ and noise $\sigma(t)$ schedulers are defined with respect to $a(t)$ and $b(t)$ as follows:

$$
s(t) = a(t), \quad \sigma(t) = \frac{b(t)}{a(t)} .

$$

#### NOTE
This SDE must be solved going reverse in time i.e. from $t=1$ to $t=0$.
Note that in order to unify flow matching and diffusion models, we set the starting time of the generating process (noise distribution) to be 1, and the ending time of the generating process (data distribution) to be 0, which is different from the convention in the flow matching literature.

* **Parameters:**
  * **a_t** (*Callable*) – time-dependent parameter $a(t)$ of flow-matching. Default to `lambda t: 1-t`.
  * **a_prime_t** (*Callable*) – time derivative $a'(t)$ of $a(t)$. Default to `lambda t: -1`.
  * **b_t** (*Callable*) – time-dependent parameter $b(t)$ of flow-matching.Default to `lambda t: t`.
  * **b_prime_t** (*Callable*) – time derivative $b'(t)$ of $b(t)$. Default to `lambda t: 1`.
  * **denoiser** ([*deepinv.models.Denoiser*](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)) – a denoiser used to provide an approximation of the score at time $t$: $\nabla \log p_t$.
  * **alpha** (*Callable* *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function $\alpha(t) = 0$ corresponds to ODE sampling and $\alpha(t) > 0$ corresponds to SDE sampling.
  * **solver** ([*deepinv.sampling.BaseSDESolver*](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver)) – the solver for solving the SDE.
  * **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) – data type of the computation, except for the `denoiser` which will use `torch.float32`.
    We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
    most computation cost is from evaluating the `denoiser`, which will be always computed in `torch.float32`.
  * **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device on which the computation is performed.
  * **\*args** – additional arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).
  * **\*\*kwargs** – additional keyword arguments for the [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).

#### velocity(x, t, \*args, \*\*kwargs)

Computes the velocity field of the flow matching process, which is defined as the drift of the backward SDE.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – current state
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – current timestep
  * **\*args** – additional arguments for the `denoiser`.
  * **\*\*kwargs** – additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.
* **Returns:**
  the velocity field at state `x` and time `t`.
* **Return type:**
  [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

<a id="sphx-glr-backref-deepinv-sampling-flowmatching"></a>

## Examples using `FlowMatching`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to perform unconditional image generation and posterior sampling using Flow Matching (FM).">  <div class="sphx-glr-thumbnail-title">Flow-Matching for posterior sampling and unconditional generation</div>
</div>
<!-- thumbnail-parent-div-close --></div>
