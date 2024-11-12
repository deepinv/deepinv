deepinv.sampling
================

This package contains posterior sampling algorithms.

Diffusion
---------
We provide various sota diffusion methods for sampling from the posterior distribution.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.DDRM
    deepinv.sampling.DiffPIR
    deepinv.sampling.DPS
    deepinv.sampling.DiffusionSampler

Markov Chain Monte Carlo Langevin
-------------------------------------
We also provide MCMC methods for sampling from the posterior distribution based on the unadjusted Langevin algorithm.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.MonteCarlo
    deepinv.sampling.ULA
    deepinv.sampling.SKRock

