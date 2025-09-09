deepinv.sampling
================

This module contains various posterior sampling algorithms, including diffusion-based methods and MCMC methods.
Please refer to the :ref:`user guide <sampling>` for more details.

Diffusion models with Stochastic Differential Equations for Image Generation and Posterior Sampling
---------------------------------------------------------------------------------------------------
.. userguide:: diffusion

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:
    
    deepinv.sampling.BaseSDE
    deepinv.sampling.DiffusionSDE
    deepinv.sampling.VarianceExplodingDiffusion
    deepinv.sampling.VariancePreservingDiffusion
    deepinv.sampling.PosteriorDiffusion
    deepinv.sampling.NoisyDataFidelity
    deepinv.sampling.DPSDataFidelity
    deepinv.sampling.BaseSDESolver
    deepinv.sampling.EulerSolver
    deepinv.sampling.HeunSolver
    deepinv.sampling.SDEOutput


Custom diffusion posterior samplers
-----------------------------------

.. userguide:: diffusion_custom

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.DDRM
    deepinv.sampling.DiffPIR
    deepinv.sampling.DPS
    deepinv.sampling.DiffusionSampler

Base Class
----------
.. userguide:: sampling

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.sampling.sampling_builder

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.sampling.BaseSampling


Markov Chain Monte Carlo Langevin
---------------------------------
.. userguide:: mcmc

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.ULA
    deepinv.sampling.SKRock

Iterators
---------
.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

    deepinv.sampling.SamplingIterator
    deepinv.sampling.SKRockIterator
    deepinv.sampling.ULAIterator
    deepinv.sampling.DiffusionIterator
