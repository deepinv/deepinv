deepinv.distributed
===================

This module provides a simplified API for distributing DeepInverse objects across
multiple devices and processes. The core function :func:`~deepinv.distributed.distribute` automatically
wraps your objects (stacked physics, denoisers, data fidelity) into their
distributed counterparts, handling all the boilerplate for you.

See the :ref:`user guide on distributed reconstruction<distributed>` for more information.

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

These classes are created automatically by :func:`deepinv.distributed.distribute`.
You typically don't need to instantiate them directly.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distributed.framework.DistributedStackedPhysics
   deepinv.distributed.framework.DistributedStackedLinearPhysics
   deepinv.distributed.framework.DistributedProcessing
   deepinv.distributed.framework.DistributedDataFidelity
   deepinv.distributed.framework.DistributedReplicatedParameters


Distribution Strategies
-----------------------

Advanced: Custom tiling strategies for spatial distribution of denoisers.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distributed.strategies.DistributedSignalStrategy
   deepinv.distributed.strategies.distributed_strategies.OverlapTilingStrategy


.. autosummary::
   :toctree: stubs
   :template: myfunc_template.rst
   :nosignatures:

   deepinv.distributed.strategies.create_strategy
