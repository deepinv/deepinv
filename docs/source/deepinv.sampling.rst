Sampling
===============================

This package contains algorithms that can obtains samples of the posterior distribution

.. math::

    - \log p(x|y,A) \propto f(y,A(x))+p(x)

where :math:`x` is the image to be reconstructed, :math:`y` are the measurements,
:math:`f(y,A(x))` is the negative log-likelihood and :math:`p(x)` is the negative log-prior.


The negative log likelihood can be set using :meth:`deepinv.optim.DataFidelity`, which includes Gaussian noise,
Poisson noise, etc. The negative log prior can be approximated using :meth:`deepinv.models.ScoreDenoiser`,
which leverages Tweedie's formula, i.e.,

.. math::

    - \nabla \log p_{\sigma}(x) \propto \left(x-D(x,\sigma)\right)/\sigma^2

where :math:`p_{\sigma} = p*\mathcal{N}(0,I\sigma^2)` is the prior convolved with a Gaussian kernel,
:math:`D(\cdot,\sigma)` is a (trained or model-based) denoiser with noise level :math:`\sigma`,
which is typically set to a low value.

.. note::

    The approximation of the prior obtained via
    :meth:`deepinv.models.ScoreDenoiser` is also valid for maximum-a-posteriori (MAP) denoisers,
    but :math:`p_{\sigma}(x)` is not given by the convolution with a Gaussian kernel, but rather
    given by the Moreau-Yosida envelope of :math:`p(x)`, i.e.,

    .. math::

        p_{\sigma}(x)=e^{- \inf_z \left(-\log p(z) + \frac{1}{2\sigma}\|x-z\|^2 \right)}.




Markov Chain Monte Carlo
--------------------------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.MCMC
    deepinv.sampling.ULA
    deepinv.sampling.SKRock


Diffusion
--------------------------------

We are currently working on adding diffusion methods to the library.