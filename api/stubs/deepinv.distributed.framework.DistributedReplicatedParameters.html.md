# DistributedReplicatedParameters

### *class* deepinv.distributed.framework.DistributedReplicatedParameters(ctx, parameters, average=True)

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Synchronize gradients for replicated trainable parameters.

This class targets parameters that are replicated on all ranks (e.g. trainable
step sizes in unrolled algorithms) and are not otherwise handled by
[`DistributedProcessing`](https://deepinv.org/api/stubs/deepinv.distributed.framework.DistributedProcessing.html.md#deepinv.distributed.framework.DistributedProcessing).
