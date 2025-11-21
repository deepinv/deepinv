deepinv.distrib
===============

This module provides a distributed computing framework for large-scale image reconstruction problems.
It enables efficient parallel processing across multiple devices or nodes by distributing measurements,
physics operators, and computations.

.. note::
   The distributed framework is designed for problems where measurements or signals are too large
   to fit in a single device's memory, or where parallel processing can significantly speed up reconstruction.


Core Components
---------------

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distrib.DistributedContext
   deepinv.distrib.DistributedPhysics
   deepinv.distrib.DistributedLinearPhysics
   deepinv.distrib.DistributedMeasurements
   deepinv.distrib.DistributedSignal
   deepinv.distrib.DistributedDataFidelity
   deepinv.distrib.DistributedPrior


Factory API
-----------

The factory API provides a simplified, configuration-driven approach to creating distributed components.
It reduces boilerplate code while keeping users in full control of their objects.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distrib.FactoryConfig
   deepinv.distrib.TilingConfig
   deepinv.distrib.DistributedBundle


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.distrib.make_distrib_bundle


Distribution Strategies
-----------------------

Strategies define how signals are split, batched, and reduced for distributed processing.
Custom strategies can be implemented by subclassing ``DistributedSignalStrategy``.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distrib.distribution_strategies.strategies.DistributedSignalStrategy
   deepinv.distrib.distribution_strategies.strategies.BasicStrategy
   deepinv.distrib.distribution_strategies.strategies.SmartTilingStrategy


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.distrib.distribution_strategies.strategies.create_strategy
