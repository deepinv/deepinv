# ScoreLoss

### *class* deepinv.loss.ScoreLoss(noise_model=None, total_batches=1000, delta=(0.001, 0.1))

Bases: [`Loss`](https://deepinv.org/api/stubs/deepinv.loss.Loss.html.md#deepinv.loss.Loss)

Learns score of distribution in the context of Noise2Score.

Proposed by Kim and Ye<sup>[1](#footcite-kim2021noise2score)</sup>.

Approximates the score of the measurement distribution $S(y)\approx \nabla \log p(y)$.

The score loss is defined as

$$
\| \epsilon + \sigma S(y+ \sigma \epsilon) \|^2
$$

where $y$ is the noisy measurement,
$S$ is the model approximating the score of the noisy measurement distribution $\nabla \log p(y)$,
$\epsilon$ is sampled from $N(0,I)$ and
$\sigma$ is sampled from $N(0,I\delta^2)$ with $\delta$ annealed during training
from a maximum value to a minimum value.

At test/evaluation time, the method uses Tweedie’s formula to estimate the score,
which depends on the noise model used:

- Gaussian noise: $R(y) = y + \sigma^2 S(y)$
- Poisson noise: $R(y) = y + \gamma y S(y)$
- Gamma noise: $R(y) = \frac{\ell y}{(\ell-1)-y S(y)}$

#### WARNING
The user should provide a backbone model $S$
to [`adapt_model`](#deepinv.loss.ScoreLoss.adapt_model) which returns the full reconstruction network
$R$, which is mandatory to compute the loss properly.

#### WARNING
This class uses the inference formula for the Poisson noise case
which differs from the one proposed in Noise2Score.

#### NOTE
This class does not support general inverse problems, it is only designed for denoising problems.

* **Parameters:**
  * **noise_model** (*None* *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Noise distribution corrupting the measurements
    (see [the physics docs](https://deepinv.org/user_guide/physics/physics.html.md#physics)). Options are [`deepinv.physics.GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise),
    [`deepinv.physics.PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise), [`deepinv.physics.GammaNoise`](https://deepinv.org/api/stubs/deepinv.physics.GammaNoise.html.md#deepinv.physics.GammaNoise) and
    [`deepinv.physics.UniformGaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.UniformGaussianNoise.html.md#deepinv.physics.UniformGaussianNoise). By default, it uses the noise model associated with
    the physics operator provided in the forward method.
  * **total_batches** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Total number of training batches (epochs \* number of batches per epoch).
  * **delta** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Tuple of two floats representing the minimum and maximum noise level,
    which are annealed during training.

<hr />

* **Example:**
  ```pycon
  >>> import torch
  >>> import deepinv as dinv
  >>> sigma = 0.1
  >>> physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
  >>> model = dinv.models.DnCNN(depth=2, pretrained=None)
  >>> loss = dinv.loss.ScoreLoss(total_batches=1, delta=(0.001, 0.1))
  >>> model = loss.adapt_model(model) # important step!
  >>> x = torch.ones((1, 3, 5, 5))
  >>> y = physics(x)
  >>> x_net = model(y, physics, update_parameters=True) # save score loss in forward
  >>> l = loss(model)
  >>> print(l.item() > 0)
  True
  ```

<hr />

* **References:**

* <a id='footcite-kim2021noise2score'>**[1]**</a> Kwanyoung Kim and Jong Chul Ye. Noise2score: tweedie’s approach to self-supervised image denoising without clean images. *Advances in Neural Information Processing Systems*, 34:864–874, 2021.

#### *class* ScoreModel(model, noise_model, delta, total_batches)

Bases: [`Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor)

Score model for the ScoreLoss.

* **Parameters:**
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Backbone model approximating the score.
  * **noise_model** (*None* *,* [*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Noise distribution corrupting the measurements
    (see [the physics docs](https://deepinv.org/user_guide/physics/physics.html.md#physics)). Options are [`deepinv.physics.GaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.GaussianNoise.html.md#deepinv.physics.GaussianNoise),
    [`deepinv.physics.PoissonNoise`](https://deepinv.org/api/stubs/deepinv.physics.PoissonNoise.html.md#deepinv.physics.PoissonNoise), [`deepinv.physics.GammaNoise`](https://deepinv.org/api/stubs/deepinv.physics.GammaNoise.html.md#deepinv.physics.GammaNoise) and
    [`deepinv.physics.UniformGaussianNoise`](https://deepinv.org/api/stubs/deepinv.physics.UniformGaussianNoise.html.md#deepinv.physics.UniformGaussianNoise). By default, it uses the noise model associated with
    the physics operator provided in the forward method.
  * **delta** ([*tuple*](https://docs.python.org/3.9/library/stdtypes.html#tuple)) – Tuple of two floats representing the minimum and maximum noise level,
    which are annealed during training.
  * **total_batches** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – Total number of training batches (epochs \* number of batches per epoch).

#### forward(y, physics, update_parameters=False, \*\*kwargs)

Computes the reconstruction of the noisy measurements.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **update_parameters** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – If True, updates the parameters of the model.

#### adapt_model(model, \*\*kwargs)

Transforms score backbone net $S$ into $R$ for training and evaluation.

* **Parameters:**
  **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Backbone model approximating the score.
* **Returns:**
  [`deepinv.loss.ScoreLoss.ScoreModel`](#deepinv.loss.ScoreLoss.ScoreModel) adapted reconstruction model.

#### forward(model, \*\*kwargs)

Computes the Score Loss.

* **Parameters:**
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – Measurements.
  * **physics** ([*deepinv.physics.Physics*](https://deepinv.org/api/stubs/deepinv.physics.Physics.html.md#deepinv.physics.Physics)) – Forward operator associated with the measurements.
  * **model** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – Reconstruction model.
* **Returns:**
  ([`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) Score loss.
