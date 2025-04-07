.. _sampling:

Diffusion and MCMC Algorithms
=============================

This package contains posterior sampling algorithms, based on diffusion models and Markov Chain Monte Carlo (MCMC) methods.

These methods build a Markov chain

.. math::

     x_{t+1} \sim p(x_{t+1} | x_t, y)

such that the samples :math:`x_t` for large :math:`t` are approximately sampled according to the posterior distribution :math:`p(x|y)`.

.. _diffusion:

Diffusion models
----------------

We provide a unified framework for image generation using diffusion models.
Diffusion models for posterior sampling are defined using :class:`deepinv.sampling.PosteriorDiffusion`,
which is a subclass of :class:`deepinv.models.base.Reconstructor`.
Below, we explain the main components of the diffusion models, see :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` for an example usage and visualizations.

Stochastic Differential Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define diffusion models as Stochastic Differential Equations (SDEs).

The **forward-time SDE** is defined as follows, from time :math:`0` to :math:`T`:

.. math::

    d\, x_t = f(t) x_t d\,t + g(t) d\, \omega_t.

Let :math:`p_t` denote the distribution of the random vector :math:`x_t`.
Under this forward process, we have that

.. math::

    x_t \vert x_0 \sim \mathcal{N}\left( s(t) x_0, \frac{\sigma(t)^2}{s(t)^2} \mathrm{I} \right),

where the **scaling** over time is :math:`s(t) = \exp\left( \int_0^t f(r) d\,r \right)` and
the **normalized noise level** is :math:`\sigma(t) = \sqrt{\int_0^t \frac{g(r)^2}{s(r)^2} d\,r}`.

The **reverse-time SDE** is defined as follows, running backwards in time (from :math:`T` to :math:`0`)

.. math::

    d\, x_{t} = \left( f(t) x_t - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_{t}(x_t) \right) d\,t + g(t) \sqrt{\alpha} d\, \omega_{t}.

where :math:`\alpha \in [0,1]` is a scalar weighting the diffusion term (:math:`\alpha = 0` corresponds to the ordinary differential equation (ODE) sampling
and :math:`\alpha > 0` corresponds to the SDE sampling), and :math:`\nabla \log p_{t}(x_t)` is the score function that can be approximated by (a properly scaled version of)
Tweedie's formula:

.. math::

    \nabla \log p_t(x_t) =  \frac{\left( \mathbb{E}\{s(t)x_0|x_t\} -  x_t \right)}{s(t)^2\sigma(t)^2} \approx \frac{\left(s(t) \denoiser{\frac{x_t}{s(t)}}{\sigma(t)} -  x_t \right)}{s(t)^2\sigma(t)^2}.

where :math:`\denoiser{\cdot}{\sigma}` is a denoiser trained to denoise images with noise level :math:`\sigma`
that is :math:`\denoiser{x+\sigma\omega}{\sigma} \approx \mathbb{E}\{ x|x+\sigma\omega\}` with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

.. note::

    Using a normalized noise levels :math:`\sigma(t)` and scalings :math:`s(t)` lets us use `any denoiser in the library <denoisers>`_
    trained for multiple noise levels assuming pixel values are in the range :math:`[0,1]`.

Starting from a random point following the end-point distribution :math:`p_T` of the forward process, 
solving the reverse-time SDE gives us a sample of the data distribution :math:`p_0`.

The base classes for defining a SDEs are :class:`deepinv.sampling.BaseSDE` and :class:`deepinv.sampling.DiffusionSDE`.

.. list-table:: Stochastic Differential Equations
   :header-rows: 1

   * - **SDE**
     - :math:`f(t)`
     - :math:`g(t)`
     - Scaling :math:`s(t)`
     - Noise level :math:`\sigma(t)`

   * - :class:`Variance Exploding <deepinv.sampling.VarianceExplodingDiffusion>`
     - :math:`0`
     - :math:`\sigma_{\mathrm{min}}\left(\frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right)^t`
     - :math:`\sigma_{\mathrm{min}}\left(\frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right)^t`
     - :math:`1`

   * - :class:`Variance Preserving <deepinv.sampling.VariancePreservingDiffusion>`
     - :math:`-\frac{1}{2}\left(\beta_{\mathrm{min}}  + t \beta_d \right)`
     - :math:`\sqrt{\beta_{\mathrm{min}}  + t \beta_{d}}`
     - :math:`1/\sqrt{e^{\frac{1}{2}\beta_{d}t^2+\beta_{\mathrm{min}}t}}`
     - :math:`\sqrt{e^{\frac{1}{2}\beta_{d}t^2+\beta_{\mathrm{min}}t}-1}`

Solvers
~~~~~~~

Once the SDE is defined, we can obtain an approximate sample with any of the following solvers:

.. list-table:: SDE/ODE solvers
   :header-rows: 1

   * - **Method**
     - **Description**

   * - :class:`deepinv.sampling.EulerSolver`
     - `First order Euler solver <https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method>`_

   * - :class:`deepinv.sampling.HeunSolver`
     - `Second order Heun solver <https://en.wikipedia.org/wiki/Heun%27s_method>`_


The base class for solvers is :class:`deepinv.sampling.BaseSDESolver`, and :class:`deepinv.sampling.SDEOutput`
provides a container for storing the output of the solver.


Posterior sampling
~~~~~~~~~~~~~~~~~~

In the case of posterior sampling, we need simply to replace the (unconditional) score function :math:`\nabla \log p_t(x_t)`
by the conditional score function :math:`\nabla \log p_t(x_t|y)`. The conditional score can be decomposed using the Bayes' rule:

.. math::

    \nabla \log p_t(x_t | y) = \nabla \log p_t(x_t) + \nabla \log p_t \left(y | \frac{x_t}{s(t)} = x_0 + \sigma(t)\omega\right).

The first term is the unconditional score function and can be approximated by using a denoiser as explained previously. 
The second term is the conditional score function, and can be approximated by the (noisy) data-fidelity term.
We implement various data-fidelity terms in :class:`deepinv.sampling.NoisyDataFidelity`.


.. list-table:: Noisy data-fidelity terms
   :header-rows: 1

   * - **Method**
     - **Description**

   * - :class:`deepinv.sampling.NoisyDataFidelity`
     - The base class for defining the noisy data-fidelity term, used to estimate the conditional score in the posterior sampling with SDE.
     
   * - :class:`deepinv.sampling.DPSDataFidelity`
     - The noisy data-fidelity term for the `Diffusion Posterior Sampling (DPS) method <https://arxiv.org/abs/2209.14687>`_. See also :class:`deepinv.sampling.DPS`.


.. _diffusion_custom:

Popular posterior samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide custom implementations of some popular diffusion methods for posterior sampling,
which can be used directly without the need to define the SDE and the solvers.

.. list-table:: Popular diffusion methods
   :header-rows: 1

   * - **Method**
     - **Description**
     - **Limitations**

   * - :class:`deepinv.sampling.DDRM`
     - Diffusion Denoising Restoration Models
     - Only for :class:`SVD decomposable operators <deepinv.physics.DecomposablePhysics>`.

   * - :class:`deepinv.sampling.DiffPIR`
     - Diffusion PnP Image Restoration
     - Only for :class:`linear operators <deepinv.physics.LinearPhysics>`.

   * - :class:`deepinv.sampling.DPS`
     - Diffusion Posterior Sampling
     - Can be slow, requires backpropagation through the denoiser.


Uncertainty quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Diffusion methods obtain a single sample per call. If multiple samples are required, the
:class:`deepinv.sampling.DiffusionSampler` can be used to convert a diffusion method into a sampler that
obtains multiple samples to compute posterior statistics such as the mean or variance.

.. _mcmc:

Markov Chain Monte Carlo
------------------------
Markov Chain Monte Carlo  (MCMC) methods build a chain of samples which aim at sampling the negative-log-posterior distribution:

.. math::

    -\log p(x|y,A) \propto d(Ax,y) + \reg{x},

where :math:`x` is the image to be reconstructed, :math:`y` are the measurements,
:math:`d(Ax,y) \propto - \log p(y|x,A)` is the negative log-likelihood and :math:`\reg{x}  \propto - \log p_{\sigma}(x)`
is the negative log-prior.

The negative log likelihood can be chosen from :ref:`this list <data-fidelity>`, and the negative log prior can be approximated using :class:`deepinv.optim.ScorePrior` with a
:ref:`pretrained denoiser <denoisers>`, which leverages Tweedie's formula with :math:`\sigma` is typically set to a small value.
Unlike diffusion sampling methods, MCMC methods generally use a fixed noise level :math:`\sigma` during the sampling process, i.e.,
:math:`\nabla \log p_t(x_t) = \frac{\left(\denoiser{x_t}{\sigma} -  x_t \right)}{\sigma^2}`.

.. note::

    The approximation of the prior obtained via
    :class:`deepinv.optim.ScorePrior` is also valid for maximum-a-posteriori (MAP) denoisers,
    but :math:`p_{\sigma}(x)` is not given by the convolution with a Gaussian kernel, but rather
    given by the Moreau-Yosida envelope of :math:`p(x)`, i.e.,

    .. math::

        p_{\sigma}(x)=e^{- \inf_z \left(-\log p(z) + \frac{1}{2\sigma}\|x-z\|^2 \right)}.


All MCMC methods inherit from :class:`deepinv.sampling.MonteCarlo`.
We also provide MCMC methods for sampling from the posterior distribution based on the unadjusted Langevin algorithm.


.. list-table:: MCMC methods
   :header-rows: 1

   * - **Method**
     - **Description**

   * - :class:`deepinv.sampling.ULA`
     - Unadjusted Langevin algorithm.

   * - :class:`deepinv.sampling.SKRock`
     - Runge-Kutta-Chebyshev stochastic approximation to accelerate the standard Unadjusted Langevin Algorithm.
