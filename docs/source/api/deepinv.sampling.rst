deepinv.sampling
================

This package contains various posterior sampling algorithms, including diffusion-based methods and MCMC methods.
Please refer to the :ref:`user guide <sampling>` for more details.

Diffusion
---------
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

    deepinv.sampling.SKRockIterator
    deepinv.sampling.ULAIterator
    deepinv.sampling.sampling_iterators.SamplingIterator
    deepinv.sampling.BaseSample

