# deepinv.distributed

This module provides a simplified API for distributing DeepInverse objects across
multiple devices and processes. The core function [`distribute()`](https://deepinv.org/api/stubs/deepinv.distributed.distribute.html.md#deepinv.distributed.distribute) automatically
wraps your objects (stacked physics, denoisers, data fidelity) into their
distributed counterparts, handling all the boilerplate for you.

See the [user guide on distributed reconstruction](https://deepinv.org/user_guide/distributed/reconstruction.html.md#distributed-reconstruction) for more information.

## Main API

These are the main components most users need:

| [`deepinv.distributed.DistributedContext`](https://deepinv.org/api/stubs/deepinv.distributed.DistributedContext.html.md#deepinv.distributed.DistributedContext)   | Context manager for distributed computing.   |
|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|

| [`deepinv.distributed.distribute`](https://deepinv.org/api/stubs/deepinv.distributed.distribute.html.md#deepinv.distributed.distribute)   | Distribute a DeepInverse object across multiple devices.   |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|

## Core Classes

These classes are created automatically by [`deepinv.distributed.distribute()`](https://deepinv.org/api/stubs/deepinv.distributed.distribute.html.md#deepinv.distributed.distribute).
You typically don’t need to instantiate them directly.

| [`deepinv.distributed.framework.DistributedStackedPhysics`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedPhysics.html.md#deepinv.distributed.framework.DistributedStackedPhysics)             | This class distributes a *collection* of physics operators across multiple processes, where each process owns a subset of the operators.   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [`deepinv.distributed.framework.DistributedStackedLinearPhysics`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedStackedLinearPhysics.html.md#deepinv.distributed.framework.DistributedStackedLinearPhysics) | Distributed linear physics operators.                                                                                                      |
| [`deepinv.distributed.framework.DistributedProcessing`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedProcessing.html.md#deepinv.distributed.framework.DistributedProcessing)                     | Distributed signal processing using pluggable tiling and reduction strategies.                                                             |
| [`deepinv.distributed.framework.DistributedDataFidelity`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedDataFidelity.html.md#deepinv.distributed.framework.DistributedDataFidelity)                 | Distributed data fidelity term for use with distributed physics operators.                                                                 |
| [`deepinv.distributed.framework.DistributedReplicatedParameters`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedReplicatedParameters.html.md#deepinv.distributed.framework.DistributedReplicatedParameters) | Synchronize gradients for replicated trainable parameters.                                                                                 |

## Distribution Strategies

Advanced: Custom tiling strategies for spatial distribution of denoisers.

| [`deepinv.distributed.strategies.DistributedSignalStrategy`](https://deepinv.org/api/stubs/deepinv.distributed.strategies.DistributedSignalStrategy.html.md#deepinv.distributed.strategies.DistributedSignalStrategy)                         | Abstract base class for distributed signal processing strategies.   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [`deepinv.distributed.strategies.distributed_strategies.TilingStrategy`](https://deepinv.org/api/stubs/deepinv.distributed.strategies.distributed_strategies.TilingStrategy.html.md#deepinv.distributed.strategies.distributed_strategies.TilingStrategy) | Smart tiling strategy with padding for N-dimensional data.          |

| [`deepinv.distributed.strategies.create_strategy`](https://deepinv.org/api/stubs/deepinv.distributed.strategies.create_strategy.html.md#deepinv.distributed.strategies.create_strategy)   | Create a distributed signal strategy.   |
|--------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
