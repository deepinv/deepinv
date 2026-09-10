<a id="sampling"></a>

# Diffusion and MCMC Algorithms

This module contains posterior sampling algorithms, based on diffusion models and Markov Chain Monte Carlo (MCMC) methods.

These methods build a Markov chain

$$
x_{t+1} \sim p(x_{t+1} | x_t, y)
$$

such that the samples $x_t$ for large $t$ are approximately sampled according to the posterior distribution $p(x|y)$.

<a id="diffusion"></a>

## Diffusion models

We provide a unified framework for image generation using diffusion models.
Diffusion models for posterior sampling are defined using [`deepinv.sampling.PosteriorDiffusion`](https://deepinv.org/api/stubs/deepinv.sampling.PosteriorDiffusion.html.md#deepinv.sampling.PosteriorDiffusion),
which is a subclass of [`deepinv.models.Reconstructor`](https://deepinv.org/api/stubs/deepinv.models.Reconstructor.html.md#deepinv.models.Reconstructor).
Below, we explain the main components of the diffusion models, see [Building your diffusion posterior sampling method using SDEs](https://deepinv.org/auto_examples/sampling/demo_diffusion_sde.html.md#sphx-glr-auto-examples-sampling-demo-diffusion-sde-py) for an example usage and visualizations.

### Stochastic Differential Equations

We define diffusion models as Stochastic Differential Equations (SDEs).

The **forward-time SDE** is defined as follows, from time $0$ to $T$:

$$
d x_t = f(t) x_t dt + g(t) d w_t.
$$

where $w_t$ is a Brownian process.
Let $p_t$ denote the distribution of the random vector $x_t$.
Under this forward process, we have that:

$$
x_t \vert x_0 \sim \mathcal{N} \left( s(t) x_0, \left( s(t)\sigma(t) \right)^2 \mathrm{I} \right),
$$

where the **scaling** over time is $s(t) = \exp\left( \int_0^t f(r) d\,r \right)$ and
the **normalized noise level** is $\sigma(t) = \sqrt{\int_0^t \frac{g(r)^2}{s(r)^2} d\,r}$.

The **reverse-time SDE** is defined as follows, running backwards in time (from $T$ to $0$):

$$
d x_{t} = \left( f(t) x_t - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_{t}(x_t) \right) dt + g(t) \sqrt{\alpha} d w_{t}.
$$

where $\alpha \in [0,1]$ is a scalar weighting the diffusion term ($\alpha = 0$ corresponds to the ordinary differential equation (ODE) sampling
and $\alpha > 0$ corresponds to the SDE sampling), and $\nabla \log p_{t}(x_t)$ is the score function that can be approximated by (a properly scaled version of)
Tweedie’s formula:

$$
\nabla \log p_t(x_t) =  \frac{  s(t) \mathbb{E}\left[x_0|x_t \right] -  x_t }{s(t)^2\sigma(t)^2} \approx \frac{s(t) \denoiser{\frac{x_t}{s(t)}}{\sigma(t)} -  x_t }{s(t)^2\sigma(t)^2}.
$$

where $\denoiser{\cdot}{\sigma}$ is a denoiser trained to denoise images with noise level $\sigma$
that is $\denoiser{x+\sigma\omega}{\sigma} \approx \mathbb{E} [ x|x+\sigma\omega ]$ with $\omega\sim\mathcal{N}(0,\mathrm{I})$.

#### NOTE
Using a normalized noise level $\sigma(t)$ and scaling $s(t)$ lets us use [any denoiser in the library](denoisers)
trained for multiple noise levels assuming pixel values are in the range $[0,1]$.

Starting from a random point following the end-point distribution $p_T$ of the forward process,
solving the reverse-time SDE gives us a sample of the data distribution $p_0$.

The base classes for defining a SDEs are [`deepinv.sampling.BaseSDE`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDE.html.md#deepinv.sampling.BaseSDE) and [`deepinv.sampling.DiffusionSDE`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSDE.html.md#deepinv.sampling.DiffusionSDE).

#### Stochastic Differential Equations

| **SDE**                                                                                                           | $f(t)$                                                        | $g(t)$                                                                                       | Scaling $s(t)$                                               | Noise level $\sigma(t)$                                                                   |
|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`Variance Exploding`](https://deepinv.org/api/stubs/deepinv.sampling.VarianceExplodingDiffusion.html.md#deepinv.sampling.VarianceExplodingDiffusion)   | $0$                                                           | $\sigma(t) \sqrt{2 \log \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)}$ | $1$                                                          | $\sigma_{\mathrm{min}}\left(\frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right)^t$ |
| [`Variance Preserving`](https://deepinv.org/api/stubs/deepinv.sampling.VariancePreservingDiffusion.html.md#deepinv.sampling.VariancePreservingDiffusion) | $-\frac{1}{2}\left(\beta_{\mathrm{min}}  + t \beta_d \right)$ | $\sqrt{\beta_{\mathrm{min}}  + t \beta_{d}}$                                                 | $1/\sqrt{e^{\frac{1}{2}\beta_{d}t^2+\beta_{\mathrm{min}}t}}$ | $\sqrt{e^{\frac{1}{2}\beta_{d}t^2+\beta_{\mathrm{min}}t}-1}$                              |

### Solvers

Once the SDE is defined, we can obtain an approximate sample with any of the following solvers:

<a id="sde-ode-solvers"></a>

#### SDE/ODE solvers

| **Method**                                                                                                 | **Description**                                                                         |
|------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [`deepinv.sampling.EulerSolver`](https://deepinv.org/api/stubs/deepinv.sampling.EulerSolver.html.md#deepinv.sampling.EulerSolver) | [First order Euler solver](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method) |
| [`deepinv.sampling.HeunSolver`](https://deepinv.org/api/stubs/deepinv.sampling.HeunSolver.html.md#deepinv.sampling.HeunSolver)   | [Second order Heun solver](https://en.wikipedia.org/wiki/Heun%27s_method)               |

The base class for solvers is [`deepinv.sampling.BaseSDESolver`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSDESolver.html.md#deepinv.sampling.BaseSDESolver), and [`deepinv.sampling.SDEOutput`](https://deepinv.org/api/stubs/deepinv.sampling.SDEOutput.html.md#deepinv.sampling.SDEOutput)
provides a container for storing the output of the solver.

### Posterior sampling

In the case of posterior sampling, we need simply to replace the (unconditional) score function $\nabla \log p_t(x_t)$
by the conditional score function $\nabla \log p_t(x_t|y)$. The conditional score can be decomposed using the Bayes’ rule:

$$
\nabla_{x_t} \log p_t(x_t | y) &= \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t \left(y \vert x_t \right) \\
                        &= \nabla_{x_t} \log p_t(x_t) + \frac{1}{s_t} \nabla_{\hat x_t} \log \hat p_t \left(y \vert \hat x_{t} = \frac{x_t}{s(t)} = x_0 + \sigma(t) \omega \right).

$$

where $\hat p_t$ stands for the distribution of the unscaled data $x_t / s(t)$.
The first term is the unconditional score function and can be approximated by using a denoiser as explained previously.
The second term is the conditional score function, which is untractable:

$$
\hat p_t(y | \hat x_t) = \int p(y|x_0) p(x_0 | \hat x_t = x_0 + \sigma(t) \omega) dx_0 .
$$

This likelihood term $\log \hat p_t$, that we call noisy data-fidelity term, can be approximated in different ways.
We implement the following noisy data-fidelity terms, which inherit from the [`deepinv.sampling.NoisyDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.NoisyDataFidelity.html.md#deepinv.sampling.NoisyDataFidelity) base class.

#### Noisy data-fidelity terms

| **Class**                                                                                                          | $\nabla_{\hat x_t} \log \hat p_t(y| \hat x_t = x + \sigma_t \omega)$                         |
|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| [`deepinv.sampling.DPSDataFidelity`](https://deepinv.org/api/stubs/deepinv.sampling.DPSDataFidelity.html.md#deepinv.sampling.DPSDataFidelity) | $\nabla_{\hat x_t} \frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{\hat x_t}{\sigma_t}} - y \|$ |

<a id="diffusion-custom"></a>

### Popular posterior samplers

We also provide custom implementations of some popular diffusion methods for posterior sampling,
which can be used directly without the need to define the SDE and the solvers.

#### Popular diffusion methods

| **Method**                                                                                         | **Description**                        | **Limitations**                                                                                                           |
|----------------------------------------------------------------------------------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.sampling.DDRM`](https://deepinv.org/api/stubs/deepinv.sampling.DDRM.html.md#deepinv.sampling.DDRM)       | Diffusion Denoising Restoration Models | Only for [`SVD decomposable operators`](https://deepinv.org/api/stubs/deepinv.physics.DecomposablePhysics.html.md#deepinv.physics.DecomposablePhysics). |
| [`deepinv.sampling.DiffPIR`](https://deepinv.org/api/stubs/deepinv.sampling.DiffPIR.html.md#deepinv.sampling.DiffPIR) | Diffusion PnP Image Restoration        | Only for [`linear operators`](https://deepinv.org/api/stubs/deepinv.physics.LinearPhysics.html.md#deepinv.physics.LinearPhysics).                 |
| [`deepinv.sampling.DPS`](https://deepinv.org/api/stubs/deepinv.sampling.DPS.html.md#deepinv.sampling.DPS)         | Diffusion Posterior Sampling           | Can be slow, requires backpropagation through the denoiser.                                                               |

### Uncertainty quantification

Diffusion methods obtain a single sample per call. If multiple samples are required, the
[`deepinv.sampling.DiffusionSampler`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionSampler.html.md#deepinv.sampling.DiffusionSampler) can be used to convert a diffusion method into a sampler that
obtains multiple samples to compute posterior statistics such as the mean or variance.
It uses the helper class [`deepinv.sampling.DiffusionIterator`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionIterator.html.md#deepinv.sampling.DiffusionIterator) to interface diffusion samplers with [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling).

<a id="mcmc"></a>

## Markov Chain Monte Carlo

Markov Chain Monte Carlo  (MCMC) methods build a chain of samples which aim at sampling the negative-log-posterior distribution:

$$
-\log p(x|y,A) \propto d(Ax,y) + \reg{x},
$$

where $x$ is the image to be reconstructed, $y$ are the measurements,
$d(Ax,y) \propto - \log p(y|x,A)$ is the negative log-likelihood and $\reg{x}  \propto - \log p_{\sigma}(x)$
is the negative log-prior.

The negative log likelihood can be chosen from [this list](https://deepinv.org/user_guide/reconstruction/optimization.html.md#data-fidelity), and the negative log prior can be approximated using [`deepinv.optim.ScorePrior`](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior) with a
[pretrained denoiser](https://deepinv.org/user_guide/reconstruction/denoisers.html.md#denoisers), which leverages Tweedie’s formula with $\sigma$ is typically set to a small value.
Unlike diffusion sampling methods, MCMC methods generally use a fixed noise level $\sigma$ during the sampling process, i.e.,
$\nabla \log p_t(x_t) = \frac{\left(\denoiser{x_t}{\sigma} -  x_t \right)}{\sigma^2}$.

#### NOTE
The approximation of the prior obtained via
[`deepinv.optim.ScorePrior`](https://deepinv.org/api/stubs/deepinv.optim.ScorePrior.html.md#deepinv.optim.ScorePrior) is also valid for maximum-a-posteriori (MAP) denoisers,
but $p_{\sigma}(x)$ is not given by the convolution with a Gaussian kernel, but rather
given by the Moreau-Yosida envelope of $p(x)$, i.e.,

$$
p_{\sigma}(x)=e^{- \inf_z \left(-\log p(z) + \frac{1}{2\sigma}\|x-z\|^2 \right)}.
$$

All MCMC methods inherit from [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling).
The function [`deepinv.sampling.sampling_builder()`](https://deepinv.org/api/stubs/deepinv.sampling.sampling_builder.html.md#deepinv.sampling.sampling_builder) returns an instance of [`deepinv.sampling.BaseSampling`](https://deepinv.org/api/stubs/deepinv.sampling.BaseSampling.html.md#deepinv.sampling.BaseSampling) with the
optimization algorithm of choice, either a predefined one (`"SKRock"`, `"ULA"`),
or with a user-defined one (an instance of [`deepinv.sampling.SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator)). For example, we can use ULA with a score prior:

```default
model = dinv.sampling.sampling_builder(iterator="ULA", prior=prior, data_fidelity=data_fidelity,
                                       params_algo={"step_size": step_size, "alpha": alpha, "sigma": sigma}, max_iter=max_iter)
x_hat = model(y, physics)
```

We provide a very flexible framework for MCMC algorithms, providing some predefined algorithms alongside making it easy to implement your own custom sampling algorithms.
This is achieved by creating your own sampling iterator, which involves subclassing [`deepinv.sampling.SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator). See [`deepinv.sampling.SamplingIterator`](https://deepinv.org/api/stubs/deepinv.sampling.SamplingIterator.html.md#deepinv.sampling.SamplingIterator) for a short example.

A custom iterator needs to implement two methods:

* `initialize_latent_variables(self, x_init, y, physics, data_fidelity, prior)`: This method sets up the initial state of your Markov chain. It receives the initial image estimate $x_{\text{init}}$, measurements $y$, the physics operator, data fidelity term, and prior. It should return a dictionary representing the initial state $X_0$, which must include the image as `{"x": x_init, ...}` and can include any other latent variables your sampler requires. The default (non overridden) behavior is returning `{"x":x_init}`
* `forward(self, X, y, physics, data_fidelity, prior, iteration_number, **iterator_specific_params)`: This method defines a single step of your MCMC algorithm. It takes the previous state $X$ (a dictionary containing at least the previous image `{"x": x, ...}`), measurements $y$, the data fidelity, the prior, and returns the new state $X_{next}$ (again, a dictionary including `{"x": x_next, ...}`).

Some predefined iterators are provided:

| Algorithm                                                                                     | Parameters                                                       |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`ULA`](https://deepinv.org/api/stubs/deepinv.sampling.ULAIterator.html.md#deepinv.sampling.ULAIterator)             | `"step_size"`, `"alpha"`, `"sigma"`                              |
| [`SKROCK`](https://deepinv.org/api/stubs/deepinv.sampling.SKRockIterator.html.md#deepinv.sampling.SKRockIterator)       | `"step_size"`, `"alpha"`, `"inner_iter"`, `"eta"`, `"sigma"`     |
| [`Diffusion`](https://deepinv.org/api/stubs/deepinv.sampling.DiffusionIterator.html.md#deepinv.sampling.DiffusionIterator) | No parameters, see the uncertainty quantification section above. |

Some legacy predefined classes are also provided:

#### MCMC methods

| **Method**                                                                                       | **Description**                                                                                          |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| [`deepinv.sampling.ULA`](https://deepinv.org/api/stubs/deepinv.sampling.ULA.html.md#deepinv.sampling.ULA)       | Unadjusted Langevin algorithm.                                                                           |
| [`deepinv.sampling.SKRock`](https://deepinv.org/api/stubs/deepinv.sampling.SKRock.html.md#deepinv.sampling.SKRock) | Runge-Kutta-Chebyshev stochastic approximation to accelerate the standard Unadjusted Langevin Algorithm. |
