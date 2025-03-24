deepinv.sampling
================

This package contains various posterior sampling algorithms, including diffusion-based methods and MCMC methods.
Please refer to the :ref:`user guide <sampling>` for more details.

Diffusion models with Stochastic Differential Equations for Image Generation and Posterior Sampling
---------------------------------------------------------------------------------------------------
.. userguide:: diffusion_generation

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:
    
    deepinv.sampling.BaseSDE
    deepinv.sampling.DiffusionSDE
    deepinv.sampling.VarianceExplodingDiffusion
    deepinv.sampling.PosteriorDiffusion
    deepinv.sampling.NoisyDataFidelity
    deepinv.sampling.DPSDataFidelity
    deepinv.sampling.sde_solver.BaseSDESolver
    deepinv.sampling.sde_solver.EulerSolver
    deepinv.sampling.sde_solver.HeunSolver


Diffusion models for posterior sampling
---------------------------------------

.. userguide:: diffusion

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.DDRM
    deepinv.sampling.DiffPIR
    deepinv.sampling.DPS
    deepinv.sampling.DiffusionSampler

Markov Chain Monte Carlo Langevin
---------------------------------
.. userguide:: mcmc

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.MonteCarlo
    deepinv.sampling.ULA
    deepinv.sampling.SKRock

