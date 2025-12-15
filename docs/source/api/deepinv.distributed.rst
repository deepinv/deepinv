deepinv.distributed
===================

This module provides a distributed computing framework for large-scale inverse problems.
It enables parallel processing across multiple GPUs through a two-function API:

1. :class:`~deepinv.distributed.DistributedContext` - manages distributed execution
2. :func:`~deepinv.distributed.distribute` - converts objects to distributed versions

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

   deepinv.distributed.DistributedContext

.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.distributed.distribute


Core Classes
------------

These classes are created automatically by :func:`~deepinv.distributed.distribute`.
You typically don't need to instantiate them directly.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distributed.DistributedPhysics
   deepinv.distributed.DistributedLinearPhysics
   deepinv.distributed.DistributedProcessing
   deepinv.distributed.DistributedDataFidelity


Distribution Strategies
-----------------------

Advanced: Custom tiling strategies for spatial distribution of denoisers/priors.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distributed.strategies.DistributedSignalStrategy
   deepinv.distributed.strategies.BasicStrategy
   deepinv.distributed.strategies.SmartTilingStrategy


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.distributed.strategies.create_strategy
