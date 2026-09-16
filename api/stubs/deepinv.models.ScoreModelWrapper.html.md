# ScoreModelWrapper

### *class* deepinv.models.ScoreModelWrapper(score_model=None, prediction_type='epsilon', clip_output=True, sigma_t=None, scale_t=None, sigma_inverse=None, variance_preserving=False, variance_exploding=False, T=1.0, takes_integer_time=False, n_timesteps=1000, \_was_trained_on_minus_one_one=True, device='cpu')

Bases: [`Denoiser`](https://deepinv.org/api/stubs/deepinv.models.Denoiser.html.md#deepinv.models.Denoiser)

Wraps a score model as a DeepInv Denoiser.

Given a noisy sample $x_t = s_t(x_0 + \sigma_t \varepsilon)$, where $\varepsilon \sim \mathcal{N}(0, I)$,
depending on the `prediction_type`, the input `score_model` is trained to predict, either:

> * the noise $\varepsilon$ (`prediction_type = 'epsilon'`) as typically the case for DDPM models, or
> * the denoised sample $x_0$ (`prediction_type = 'sample'`) or
> * the `v-prediction` $s_t (\varepsilon - \sigma_t \cdot x_0)$ as proposed by <sup>[1](#footcite-salimans2022progressive)</sup> (`prediction_type = 'v_prediction'`)
* **Parameters:**
  * **score_model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) *|* *Callable*) – score model to be wrapped.
  * **prediction_type** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – type of prediction made by the score model.
  * **clip_output** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether to clip the output to the model range. Default is `True`.
  * **sigma_t** (*Callable* *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – continuous function or tensor (of shape `[N]` with `N` the number of time steps) defining the noise schedule $\sigma_t$.
  * **scale_t** (*Callable* *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – function or tensor (of shape `[N]` with `N` the number of time steps) defining the scaling schedule $s_t$.
  * **sigma_inverse** (*Callable*) – analytic inverse of the `sigma_t`. If not provided, a numeric inversion is used.
  * **variance_preserving** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the schedule is variance-preserving. If `True`, `scale_t` is computed from the `sigma_t`.
  * **variance_exploding** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the schedule is variance-exploding. If `True`, `scale_t` is set to `1`.
  * **T** ([*float*](https://docs.python.org/3.9/library/functions.html#float)) – maximum time value for continuous schedules. Default is `1.0`.
  * **takes_integer_time** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the model takes integer time steps (in `[0, n_timesteps-1]`) as input. Default is `False`.
  * **n_timesteps** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – number of time steps for discrete schedules. Default is `1000`.
  * **\_was_trained_on_minus_one_one** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the model was trained on images in `[-1, 1]` range (`True`) or `[0, 1]` range (`False`). Default is `True`.
  * **str** – device to load the model on. Default is `'cpu'`.

<hr />

* **References:**

* <a id='footcite-salimans2022progressive'>**[1]**</a> Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. *arXiv preprint arXiv:2202.00512*, 2022.

#### forward(x, sigma=None, input_in_minus_one_one=False, \*args, \*\*kwargs)

Applies denoiser $\denoiser{x}{\sigma}$.
If `input_in_minus_one_one` is `False` (default value), the input `x` is expected to be in `[0, 1]` range (up to random noise) and the output is also in `[0, 1]` range.
Otherwise, both input and output are expected in `[-1, 1]` range.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – noisy input, of shape `[B, C, H, W]`.
  * **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *,* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level. Can be a `float` or a [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) of shape `[B]`.
    If a single `float` is provided, the same noise level is used for all samples in the batch.
    Otherwise, batch-wise noise levels are used.
  * **input_in_minus_one_one** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – whether the input `x` is in `[-1, 1]` range. Default is `False`.
  * **args** – additional positional arguments to be passed to the model.
  * **kwarg** – additional keyword arguments to be passed to the model. For example, a `prompt` for text-conditioned or `class_label` for class-conditioned models.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the denoised output.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### get_schedule_value(schedule, t, target_size=None)

Get the value of a schedule (function or tensor) at given time steps.

* **Parameters:**
  * **schedule** (*Callable* *|* [*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – schedule function or tensor.
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – time steps, of shape `[B]` or `[]`.
  * **target_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)) – target size to broadcast the output to. Default is `None`.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) schedule values at time steps `t`, of shape that is broadcastable to `target_size` if `target_size` is provided.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### score(x, t=None, \*args, \*\*kwargs)

Computes the score function $\nabla_x \log p_t(x)$.

* **Parameters:**
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input tensor of shape `[B, C, H, W]`.
  * **t** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *|* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – single timestep or tensor of shape `[B]` or `[]`.
  * **args** – additional positional arguments of the model.
  * **kwargs** – additional keyword arguments of the model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) the score function of shape `[B, C, H, W]`.
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### time_from_sigma(sigma)

Computes the time step `t in [0,T]` corresponding to a given noise level `sigma`.

If an analytic inverse of the `sigma_t` is provided, it is used.
Otherwise, a numeric inversion is performed (nearest neighbor for discrete schedules, binary search for continuous schedules).

* **Parameters:**
  **sigma** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) *|* [*float*](https://docs.python.org/3.9/library/functions.html#float)) – noise level(s), either a scalar or a tensor of shape `[B]`.

<a id="sphx-glr-backref-deepinv-models-scoremodelwrapper"></a>

## Examples using `ScoreModelWrapper`:

<div class="sphx-glr-thumbnails">
<!-- thumbnail-parent-div-open --><div class="sphx-glr-thumbcontainer" tooltip="This demo shows you how to use our wrapper deepinv.models.DiffusersDenoiserWrapper to turn any SOTA models from the HuggingFace Hub to an image denoiser. It also can be used to perform unconditional image generation or for posterior sampling.">  <div class="sphx-glr-thumbnail-title">Using state-of-the-art diffusion models from HuggingFace Diffusers with DeepInverse</div>
</div>
<!-- thumbnail-parent-div-close --></div>
