.. _sampling:

Diffusion and MCMC Algorithms
=============================

This package contains posterior sampling algorithms.

.. math::

    - \log p(x|y,A) \propto d(Ax,y) + \reg{x},

where :math:`x` is the image to be reconstructed, :math:`y` are the measurements,
:math:`d(Ax,y) \propto - \log p(y|x,A)` is the negative log-likelihood and :math:`\reg{x}  \propto - \log p_{\sigma}(x)`
is the negative log-prior.

.. _diffusion:

Diffusion
---------
We provide various sota diffusion methods for sampling from the posterior distribution.
Diffusion methods produce a sample from the posterior ``x`` given a
measurement ``y`` as ``x = model(y, physics)``,
where ``model`` is the diffusion algorithm and ``physics`` is the forward operator.


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

The negative log likelihood from :ref:`this list <data-fidelity>`:, which includes Gaussian noise,
Poisson noise, etc. The negative log prior can be approximated using :class:`deepinv.optim.ScorePrior` with a
:ref:`pretrained denoiser <denoisers>`, which leverages Tweedie's formula, i.e.,

.. math::

    - \nabla \log p_{\sigma}(x) \propto \left(x-\denoiser{x}{\sigma}\right)/\sigma^2

where :math:`p_{\sigma} = p*\mathcal{N}(0,I\sigma^2)` is the prior convolved with a Gaussian kernel,
:math:`\denoiser{\cdot}{\sigma}` is a (trained or model-based) denoiser with noise level :math:`\sigma`,
which is typically set to a low value.

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
