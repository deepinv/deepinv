.. _sampling:

Diffusion and MCMC Algorithms
=============================

This module contains posterior sampling algorithms, based on diffusion models and Markov Chain Monte Carlo (MCMC) methods.

These methods build a Markov chain

.. math::

     x_{t+1} \sim p(x_{t+1} | x_t, y)

such that the samples :math:`x_t` for large :math:`t` are approximately sampled according to the posterior distribution :math:`p(x|y)`.

.. _diffusion:

Diffusion models
----------------

We provide a unified framework for image generation using diffusion models.
Diffusion models for posterior sampling are defined using :class:`deepinv.sampling.PosteriorDiffusion`,
which is a subclass of :class:`deepinv.models.Reconstructor`.
Below, we explain the main components of the diffusion models, see :ref:`sphx_glr_auto_examples_sampling_demo_diffusion_sde.py` for an example usage and visualizations.

Stochastic Differential Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define diffusion models as Stochastic Differential Equations (SDEs).

The **forward-time SDE** is defined as follows, from time :math:`0` to :math:`T`:

.. math::

    d x_t = f(t) x_t dt + g(t) d w_t.

where :math:`w_t` is a Brownian process. 
Let :math:`p_t` denote the distribution of the random vector :math:`x_t`.
Under this forward process, we have that:

.. math::

    x_t \vert x_0 \sim \mathcal{N} \left( s(t) x_0, \frac{\sigma(t)^2}{s(t)^2} \mathrm{I} \right),

where the **scaling** over time is :math:`s(t) = \exp\left( \int_0^t f(r) d\,r \right)` and
the **normalized noise level** is :math:`\sigma(t) = \sqrt{\int_0^t \frac{g(r)^2}{s(r)^2} d\,r}`.

The **reverse-time SDE** is defined as follows, running backwards in time (from :math:`T` to :math:`0`):

.. math::

    d x_{t} = \left( f(t) x_t - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_{t}(x_t) \right) dt + g(t) \sqrt{\alpha} d w_{t}.

where :math:`\alpha \in [0,1]` is a scalar weighting the diffusion term (:math:`\alpha = 0` corresponds to the ordinary differential equation (ODE) sampling
and :math:`\alpha > 0` corresponds to the SDE sampling), and :math:`\nabla \log p_{t}(x_t)` is the score function that can be approximated by (a properly scaled version of)
Tweedie's formula:

.. math::

    \nabla \log p_t(x_t) =  \frac{\mathbb{E}\left[ s(t)x_0|x_t \right] -  x_t }{s(t)^2\sigma(t)^2} \approx \frac{s(t) \denoiser{\frac{x_t}{s(t)}}{\sigma(t)} -  x_t }{s(t)^2\sigma(t)^2}.

where :math:`\denoiser{\cdot}{\sigma}` is a denoiser trained to denoise images with noise level :math:`\sigma`
that is :math:`\denoiser{x+\sigma\omega}{\sigma} \approx \mathbb{E} [ x|x+\sigma\omega ]` with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

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
     - :math:`1`
     - :math:`\sigma_{\mathrm{min}}\left(\frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}}\right)^t`

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

    \nabla \log p_t(x_t | y) = \nabla \log p_t(x_t) + \nabla \log p_t \left(y \vert \frac{x_t}{s(t)} = x_0 + \sigma(t)\omega\right).

The first term is the unconditional score function and can be approximated by using a denoiser as explained previously. 
The second term is the conditional score function, and can be approximated by the (noisy) data-fidelity term.
We implement the following data-fidelity terms, which inherit from the :class:`deepinv.sampling.NoisyDataFidelity` base class.

.. list-table:: Noisy data-fidelity terms
   :header-rows: 1

   * - **Class**
     - :math:`\nabla_x \log p_t(y|x + \epsilon\sigma(t))`

   * - :class:`deepinv.sampling.DPSDataFidelity`
     - :math:`\nabla_x \frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|`


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
It uses the helper class :class:`deepinv.sampling.DiffusionIterator` to interface diffusion samplers with :class:`deepinv.sampling.BaseSampling`.

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


All MCMC methods inherit from :class:`deepinv.sampling.BaseSampling`.
The function :func:`deepinv.sampling.sampling_builder` returns an instance of :class:`deepinv.sampling.BaseSampling` with the
optimization algorithm of choice, either a predefined one (``"SKRock"``, ``"ULA"``),
or with a user-defined one (an instance of :class:`deepinv.sampling.SamplingIterator`). For example, we can use ULA with a score prior:

::

    model = dinv.sampling.sampling_builder(iterator="ULA", prior=prior, data_fidelity=data_fidelity,
                                           params_algo={"step_size": step_size, "alpha": alpha, "sigma": sigma}, max_iter=max_iter)
    x_hat = model(y, physics)


We provide a very flexible framework for MCMC algorithms, providing some predefined algorithms alongside making it easy to implement your own custom sampling algorithms.
This is achieved by creating your own sampling iterator, which involves subclassing :class:`deepinv.sampling.SamplingIterator`. See :class:`deepinv.sampling.SamplingIterator` for a short example.

A custom iterator needs to implement two methods:

*   ``initialize_latent_variables(self, x_init, y, physics, data_fidelity, prior)``: This method sets up the initial state of your Markov chain. It receives the initial image estimate :math:`x_{\text{init}}`, measurements :math:`y`, the physics operator, data fidelity term, and prior. It should return a dictionary representing the initial state :math:`X_0`, which must include the image as ``{"x": x_init, ...}`` and can include any other latent variables your sampler requires. The default (non overridden) behavior is returning ``{"x":x_init}``

*   ``forward(self, X, y, physics, data_fidelity, prior, iteration_number, **iterator_specific_params)``: This method defines a single step of your MCMC algorithm. It takes the previous state :math:`X` (a dictionary containing at least the previous image ``{"x": x, ...}``), measurements :math:`y`, the data fidelity, the prior, and returns the new state :math:`X_{next}` (again, a dictionary including ``{"x": x_next, ...}``). 


Some predefined iterators are provided:

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Parameters

   * - :class:`ULA <deepinv.sampling.ULAIterator>`
     - ``"step_size"``, ``"alpha"``, ``"sigma"``

   * - :class:`SKROCK <deepinv.sampling.SKRockIterator>`
     - ``"step_size"``, ``"alpha"``, ``"inner_iter"``, ``"eta"``, ``"sigma"``

   * - :class:`Diffusion <deepinv.sampling.DiffusionIterator>`
     - No parameters, see the uncertainty quantification section above.


Some legacy predefined classes are also provided:


.. list-table:: MCMC methods
   :header-rows: 1

   * - **Method**
     - **Description**

   * - :class:`deepinv.sampling.ULA`
     - Unadjusted Langevin algorithm.

   * - :class:`deepinv.sampling.SKRock`
     - Runge-Kutta-Chebyshev stochastic approximation to accelerate the standard Unadjusted Langevin Algorithm.
