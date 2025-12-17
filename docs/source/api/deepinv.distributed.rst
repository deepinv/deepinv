deepinv.distributed
===================

This module provides a simplified API for distributing DeepInverse objects across
multiple devices and processes. The core function :func:`~deepinv.distributed.distribute` automatically
wraps your objects (stacked physics, denoisers, data fidelity) into their
distributed counterparts, handling all the boilerplate for you.

**Main Components:**

   - :func:`~deepinv.distributed.distribute`: Universal distributor for DeepInverse objects
   - :class:`~deepinv.distributed.DistributedContext`: Manages distributed execution environment

Simply pass your DeepInverse object and a :class:`~deepinv.distributed.DistributedContext` to the
:func:`~deepinv.distributed.distribute` function. It automatically detects the object type and returns
the appropriate distributed wrapper:

   - **Stacked Physics or list of Physics** → :class:`~deepinv.distributed.DistributedPhysics` or :class:`~deepinv.distributed.DistributedLinearPhysics`
   - **Denoisers** → :class:`~deepinv.distributed.DistributedProcessing` (with spatial tiling)
   - **Data fidelity** → :class:`~deepinv.distributed.DistributedDataFidelity`

**Key Benefits:**

   - **Automatic Type Detection**: The API figures out what you're distributing
   - **Production Ready**: Handles multi-GPU, multi-node setups automatically
   - **Seamless Integration**: Works naturally with DeepInverse optimization algorithms

The returned objects work seamlessly with DeepInverse's optimization algorithms and
provide both local operations and automatic global reduction when needed.

**The distributed framework is designed for:**

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

These classes are created automatically by :func:`deepinv.distributed.distribute`.
You typically don't need to instantiate them directly.

.. autosummary::
   :toctree: stubs
   :template: myclass_template.rst
   :nosignatures:

   deepinv.distributed.DistributedStackedPhysics
   deepinv.distributed.DistributedStackedLinearPhysics
   deepinv.distributed.DistributedProcessing
   deepinv.distributed.DistributedDataFidelity


Distribution Strategies
-----------------------

Advanced: Custom tiling strategies for spatial distribution of denoisers.

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
