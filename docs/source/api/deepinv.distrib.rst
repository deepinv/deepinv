deepinv.distrib
===============

This module provides a distributed computing framework for large-scale inverse problems.
It enables parallel processing across multiple GPUs through a two-function API:

1. :class:`~deepinv.distrib.DistributedContext` - manages distributed execution
2. :func:`~deepinv.distrib.distribute` - converts objects to distributed versions

.. note::
   The distributed framework is designed for:
   
   - **Multi-operator problems**: Parallel processing of multiple physics operators
   - **Large images**: Spatial tiling for images or volumes too large for single-device memory
   - **Acceleration**: Leveraging multiple devices for faster reconstruction


Main API
--------

These are the main components most users need:

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distrib.DistributedContext

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.distrib.distribute


Core Classes
------------

These classes are created automatically by :func:`~deepinv.distrib.distribute`.
You typically don't need to instantiate them directly.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distrib.DistributedPhysics
   deepinv.distrib.DistributedLinearPhysics
   deepinv.distrib.DistributedProcessing
   deepinv.distrib.DistributedDataFidelity


Distribution Strategies
-----------------------

Advanced: Custom tiling strategies for spatial distribution of denoisers/priors.

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
