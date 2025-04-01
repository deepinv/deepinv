.. _sampling:

Diffusion and MCMC Algorithms
=============================

This package contains posterior sampling algorithms. 
The negative-log-posterior can be written as:

.. math::

    -\log p(x|y,A) \propto d(Ax,y) + \reg{x},

where :math:`x` is the image to be reconstructed, :math:`y` are the measurements,
:math:`d(Ax,y) \propto - \log p(y|x,A)` is the negative log-likelihood and :math:`\reg{x}  \propto - \log p_{\sigma}(x)`
is the negative log-prior.

.. _diffusion:

Diffusion models with Stochastic Differential Equations for Image Generation and Posterior Sampling
---------------------------------------------------------------------------------------------------
 
We first provide a unified framework for image generation using diffusion models.
We define diffusion models as Stochastic Differential Equations (SDE).

The forward-time SDE is defined as follows, for :math:`t \in [0, T]`:

.. math::

    d\, x_t = f(x_t, t) d\,t + g(t) d\, w_t.

Let :math:`p_t` denote the distribution of the random vector :math:`x_t`.
The reverse-time SDE is defined as follows, running backward in time:

.. math::

    d\, x_{t} = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_t(x_t) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}.

The scalar :math:`\alpha \in [0,1]` weighting the diffusion term. :math:`\alpha = 0` corresponds to the ordinary differential equation (ODE) sampling and :math:`\alpha > 0` corresponds to the SDE sampling.

This reverse-time SDE can be used as a generative process. 

The (Stein) score function :math:`\nabla \log p_t(x_t)` can be approximated by Tweedie's formula. In particular, if 

.. math::

    x_t \vert x_0 \sim \mathcal{N}\left( \mu_t x_0, \sigma_t^2 \mathrm{Id} \right),

then

.. math::

    \nabla \log p_t(x_t) = \frac{\left(\mu_t D_{\sigma_t}(x_t) -  x_t \right)}{\sigma_t^2}.

Starting from a random point following the end-point distribution :math:`p_T` of the forward process, 
solving the reverse-time SDE gives us a sample of the data distribution :math:`p_0`.

In the case of posterior sampling, we need simply to repace the (unconditional) score function :math:`\nabla_{x_t} \log p_t(x_t)` by the conditional score function :math:`\nabla_{x_t} \log p_t(x_t|y)`. The conditional score can be decomposed using the Bayes' rule:

.. math::
    \nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t).

The first term is the unconditional score function and can be approximated by using a denoiser as explained previously. 
The second term is the conditional score function, and can be approximated by the (noisy) data-fidelity term. We implement various data-fidelity terms in :class:`deepinv.sampling.NoisyDataFidelity`.

.. list-table:: Stochastic Differential Equations
   :header-rows: 1

   * - **Method**
     - **Description**

   * - :class:`deepinv.sampling.BaseSDE`
     - Base class for defining a SDE with a drift term and a diffusion coefficient

   * - :class:`deepinv.sampling.DiffusionSDE`
     - Define automatically the reverse-time SDE from a forward SDE and a denoiser. 

   * - :class:`deepinv.sampling.VarianceExplodingDiffusion`
     - The Variance-Exploding SDE, an instance of :meth:`deepinv.sampling.DiffusionSDE`

   * - :class:`deepinv.sampling.VariancePreservingDiffusion`
     - The Variance-Preserving SDE (corresponds to DDPM), an instance of :meth:`deepinv.sampling.DiffusionSDE`

   * - :class:`deepinv.sampling.PosteriorDiffusion`
     - The Diffusion SDE class for posterior sampling, a subclass of :class:`deepinv.models.base.Reconstructor`


.. list-table:: Noisy data-fidelity terms
   :header-rows: 1

   * - **Method**
     - **Description**

   * - :class:`deepinv.sampling.NoisyDataFidelity`
     - The base class for defining the noisy data-fidelity term, used to estimate the conditional score in the posterior sampling with SDE.
     
   * - :class:`deepinv.sampling.DPSDataFidelity`
     - The noisy data-fidelity term for the `Diffusion Posterior Sampling (DPS) method <https://arxiv.org/abs/2209.14687>`_

We also provide generic methods for solving SDEs (and ODEs).

.. list-table:: SDE/ODE solvers
   :header-rows: 1

   * - **Method**
     - **Description**
  
   * - :class:`deepinv.sampling.BaseSDESolver`
     - Base class of the solvers.

   * - :class:`deepinv.sampling.SDEOutput`
     - Container for storing the output of an SDE solver.

   * - :class:`deepinv.sampling.EulerSolver`
     - `First order Euler solver <https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method>`_ 

   * - :class:`deepinv.sampling.HeunSolver`
     - `Second order Heun solver. <https://en.wikipedia.org/wiki/Heun%27s_method>`_



.. _diffusion_custom:

Custom diffusion posterior samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide custom implementations of some popular diffusion methods for posterior sampling.
Diffusion methods obtain a single sample per call. If multiple samples are required, the
:class:`deepinv.sampling.DiffusionSampler` can be used to convert a diffusion method into a sampler that
obtains multiple samples to compute posterior statistics such as the mean or variance.

.. list-table:: Diffusion methods
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


.. _mcmc:

Markov Chain Monte Carlo
------------------------
Unlike diffusion sampling methods, the MCMC method does not change the noise level :math:`\sigma` during the sampling process.
It can be seen as a stochastic gradient method for minimizing the negative-log-posterior defined above, with a fixed value of :math:`sigma`.
The negative log likelihood from :ref:`this list <data-fidelity>`:, which includes Gaussian noise,
Poisson noise, etc. The negative log prior can be approximated using :class:`deepinv.optim.ScorePrior` with a
:ref:`pretrained denoiser <denoisers>`, which leverages Tweedie's formula with :math:`\sigma` is typically set to a small value.

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
