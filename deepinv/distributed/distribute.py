r"""
Distributed Framework API
==========================

This module provides a simplified API for distributing DeepInverse objects across
multiple devices and processes. The core function :func:`distribute` automatically
wraps your objects (physics operators, denoisers, data fidelity terms) into their
distributed counterparts, handling all the boilerplate for you.

**Main Function:**

- :func:`distribute`: Universal distributor for DeepInverse objects

**How It Works:**

Simply pass your DeepInverse object and a :class:`DistributedContext` to the
:func:`distribute` function. It automatically detects the object type and returns
the appropriate distributed wrapper:

- **Physics operators** → :class:`DistributedPhysics` or :class:`DistributedLinearPhysics`
- **Denoisers/Priors** → :class:`DistributedProcessing` (with spatial tiling)
- **Data fidelity terms** → :class:`DistributedDataFidelity`

**Key Benefits:**

- **Automatic Type Detection**: The API figures out what you're distributing
- **Production Ready**: Handles multi-GPU, multi-node setups automatically

**Quick Example:**

.. code-block:: python

    from deepinv.physics import Blur
    from deepinv.models import DnCNN
    from deepinv.optim.data_fidelity import L2
    from deepinv.distributed import DistributedContext, distribute

    # Create distributed context (detects your environment automatically)
    with DistributedContext() as ctx:
        # Distribute physics operators
        physics_list = [Blur(...), Inpainting(...), MRI(...)]
        dphysics = distribute(physics_list, ctx)

        # Distribute a denoiser with spatial tiling
        denoiser = DnCNN()
        ddenoiser = distribute(denoiser, ctx, patch_size=256, receptive_field_size=64)

        # Distribute data fidelity
        data_fidelity = L2()
        dfidelity = distribute(data_fidelity, ctx)

        # Use them naturally
        x = torch.randn(1, 3, 1024, 1024)
        y = dphysics(x)  # Distributed forward pass
        x_denoised = ddenoiser(x)  # Distributed denoising with tiling

The returned objects work seamlessly with DeepInverse's optimization algorithms and
provide both local operations and automatic global reduction when needed.
"""

from __future__ import annotations

from typing import Callable

import torch

from deepinv.physics import Physics, LinearPhysics
from deepinv.physics.forward import StackedPhysics, StackedLinearPhysics
from deepinv.optim.data_fidelity import DataFidelity, StackedPhysicsDataFidelity
from deepinv.optim.prior import Prior
from deepinv.models.base import Denoiser

from deepinv.distributed.distrib_framework import (
    DistributedContext,
    DistributedPhysics,
    DistributedLinearPhysics,
    DistributedProcessing,
    DistributedDataFidelity,
)

from deepinv.distributed.strategies import DistributedSignalStrategy


def _distribute_physics(
    physics: (
        StackedPhysics
        | list[Physics]
        | Callable[[int, torch.device, dict | None], Physics]
    ),
    ctx: DistributedContext,
    *,
    num_operators: int | None = None,
    type_object: str | None = "physics",
    dtype: torch.dtype | None = torch.float32,
    gather_strategy: str = "concatenated",
    **kwargs,
) -> DistributedPhysics | DistributedLinearPhysics:
    r"""
    Distribute a Physics object across multiple devices.

    :param Physics physics: Physics object to distribute
    :param DistributedContext ctx: distributed context manager
    :param None, int num_operators: number of physics operators when using a factory for physics, otherwise inferred.
    :param str type_object: type of physics object to distribute. Options are `'physics'` or `'linear_physics'`. Default is `'physics'`.
    :param None, torch.dtype dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
        Default is `'concatenated'`.
    :param kwargs: additional keyword arguments for DistributedPhysics

    :returns: Distributed version of the input Physics object
    """
    # Physics factory
    if isinstance(physics, (StackedPhysics, StackedLinearPhysics)):
        # Extract physics_list from StackedPhysics
        physics_list_extracted = physics.physics_list
        num_operators = len(physics_list_extracted)

        def physics_factory(idx: int, device: torch.device, shared: dict | None):
            return physics_list_extracted[idx].to(device)

    elif callable(physics):
        physics_factory = physics
        if num_operators is None or not isinstance(num_operators, int):
            raise ValueError(
                "When using a factory for physics, you must provide num_operators."
            )
    else:
        physics_list_extracted = physics
        num_operators = len(physics_list_extracted)

        def physics_factory(idx: int, device: torch.device, shared: dict | None):
            return physics_list_extracted[idx].to(device)

    if type_object == "linear_physics":
        return DistributedLinearPhysics(
            ctx,
            num_operators=num_operators,
            factory=physics_factory,
            dtype=dtype,
            gather_strategy=gather_strategy,
            **kwargs,
        )
    else:
        return DistributedPhysics(
            ctx,
            num_operators=num_operators,
            factory=physics_factory,
            dtype=dtype,
            gather_strategy=gather_strategy,
            **kwargs,
        )


def _distribute_processor(
    processor: Prior | Denoiser,
    ctx: DistributedContext,
    *,
    dtype: torch.dtype | None = torch.float32,
    tiling_strategy: torch.dtype | None = None,
    patch_size: int = 256,
    receptive_field_size: int = 64,
    tiling_dims: int | tuple[int, ...] | None = None,
    max_batch_size: int | None = None,
    gather_strategy: str = "concatenated",
    **kwargs,
) -> DistributedProcessing:
    r"""
    Distribute a DeepInverse prior or denoiser across multiple devices.

    :param Prior | Denoiser processor: DeepInverse prior or denoiser to distribute
    :param DistributedContext ctx: distributed context manager
    :param None, torch.dtype dtype: data type for distributed object. Default is `torch.float32`.
    :param str | DistributedSignalStrategy | None tiling_strategy: strategy for tiling the signal. Options are `'basic'`, `'smart_tiling'`, or a custom strategy instance. Default is `'smart_tiling'`.
    :param int patch_size: size of patches for tiling strategies. Default is 256.
    :param int receptive_field_size: receptive field size for overlap in tiling strategies. Default is 64.
    :param bool overlap: whether patches should overlap. Default is False.
    :param None, int max_batch_size: maximum number of patches to process in a single batch. If `None`, all patches are batched together. Set to 1 for sequential processing.
    :param str gather_strategy: strategy for gathering distributed results (currently unused for processors, kept for API consistency).
    :param kwargs: additional keyword arguments for DistributedProcessing

    :returns: Distributed version of the input processor
    """

    return DistributedProcessing(
        ctx=ctx,
        processor=processor,
        dtype=dtype,
        strategy=tiling_strategy,
        strategy_kwargs={
            "patch_size": patch_size,
            "receptive_field_size": receptive_field_size,
            "tiling_dims": tiling_dims,
        },
        max_batch_size=max_batch_size,
    )


def _distribute_data_fidelity(
    data_fidelity: (
        DataFidelity
        | StackedPhysicsDataFidelity
        | list[DataFidelity]
        | Callable[[int, torch.device, dict | None], DataFidelity]
    ),
    ctx: DistributedContext,
    num_operators: int | None = None,
    **kwargs,
) -> DistributedDataFidelity:
    r"""
    Distribute a DataFidelity object across multiple devices.

    :param DataFidelity data_fidelity: DataFidelity object to distribute
    :param DistributedContext ctx: distributed context manager
    :param None, int num_operators: number of data fidelity operators when using a factory for data_fidelity, otherwise inferred.
    :param kwargs: additional keyword arguments for DistributedDataFidelity

    :returns: Distributed version of the input DataFidelity object
    """
    # DataFidelity factory

    if isinstance(data_fidelity, DataFidelity):
        return DistributedDataFidelity(
            ctx,
            data_fidelity=data_fidelity,
            num_operators=num_operators,
            **kwargs,
        )

    elif isinstance(data_fidelity, StackedPhysicsDataFidelity):
        data_fidelity_list_extracted = data_fidelity.data_fidelity_list
        num_operators = len(data_fidelity_list_extracted)

        def data_fidelity_factory(idx: int, device: torch.device, shared: dict | None):
            return data_fidelity_list_extracted[idx].to(device)

    elif callable(data_fidelity):
        data_fidelity_factory = data_fidelity
        if num_operators is None or not isinstance(num_operators, int):
            raise ValueError(
                "When using a factory for data_fidelity, you must provide num_operators."
            )
    else:
        data_fidelity_list_extracted = data_fidelity
        num_operators = len(data_fidelity_list_extracted)

        def data_fidelity_factory(idx: int, device: torch.device, shared: dict | None):
            return data_fidelity_list_extracted[idx].to(device)

    return DistributedDataFidelity(
        ctx,
        data_fidelity=data_fidelity_factory,
        num_operators=num_operators,
        **kwargs,
    )


def distribute(
    object: (
        StackedPhysics
        | list[Physics]
        | Callable[[int, torch.device, dict | None], Physics]
        | Denoiser
        | Callable[[int, torch.device, dict | None], Denoiser]
        | DataFidelity
        | list[DataFidelity]
        | StackedPhysicsDataFidelity
        | Callable[[int, torch.device, dict | None], DataFidelity]
    ),
    ctx: DistributedContext,
    *,
    num_operators: int | None = None,
    type_object: str | None = "auto",
    dtype: torch.dtype | None = torch.float32,
    gather_strategy: str = "concatenated",
    tiling_strategy: str | DistributedSignalStrategy | None = None,
    tiling_dims: int | tuple[int, ...] | None = None,
    patch_size: int = 256,
    receptive_field_size: int = 64,
    max_batch_size: int | None = None,
    **kwargs,
) -> (
    DistributedPhysics
    | DistributedLinearPhysics
    | DistributedProcessing
    | DistributedDataFidelity
):
    r"""
    Distribute a DeepInverse object across multiple devices.

    This function takes a DeepInverse object (Physics, DataFidelity, or Prior)
    and distributes it using the provided DistributedContext.

    :param StackedPhysics | list[Physics] | Callable[[int, torch.device, dict | None], Physics] | Denoiser | Callable[[int, torch.device, dict | None], Denoiser] | DataFidelity | list[DataFidelity] | StackedPhysicsDataFidelity | Callable[[int, torch.device, dict | None], DataFidelity] object: DeepInverse object to distribute
    :param DistributedContext ctx: distributed context manager
    :param None, int num_operators: number of physics operators when using a factory for physics, otherwise inferred.
    :param str | None type_object: type of object to distribute. Options are `'physics'`, `'data_fidelity'`, or `'auto'` for automatic detection. Default is `'auto'`.
    :param torch.dtype | None dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
        Default is `'concatenated'`.
    :param str | DistributedSignalStrategy | None tiling_strategy: strategy for tiling the signal (for Denoiser/Prior). Options are `'basic'`, `'smart_tiling'`, or a custom strategy instance. Default is `'smart_tiling'`.
    :param int | tuple[int, ...] | None tiling_dims: dimensions to tile over (for Denoiser/Prior).
        If ``None`` (default), tiles the last N-2 dimensions (spatial dimensions).
        If an int ``N``, tiles the last ``N`` dimensions.
        If a tuple, specifies exact dimensions to tile.
        Examples:
        - For ``(B, C, H, W)`` image: ``tiling_dims=2`` tiles H and W.
        - For ``(B, C, D, H, W)`` volume: ``tiling_dims=3`` tiles D, H, W.
    :param int | tuple[int, ...] patch_size: size of patches for tiling strategies (for Denoiser/Prior).
        Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is 256.
    :param int | tuple[int, ...] receptive_field_size: receptive field size for overlap in tiling strategies (for Denoiser/Prior).
        Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is 64.
    :param None, int max_batch_size: maximum number of patches to process in a single batch (for Denoiser/Prior). If `None`, all patches are batched together. Set to 1 for sequential processing.
    :param kwargs: additional keyword arguments for specific distributed classes

    :returns: Distributed version of the input object

    |sep|

    :Examples:

        Distribute a Physics object:

        >>> from deepinv.physics import Blur, StackedLinearPhysics
        >>> from deepinv.distributed import DistributedContext, distribute
        >>> with DistributedContext() as ctx: # doctest: +SKIP
        ...     physics = StackedLinearPhysics([Blur(kernel_size=5), Blur(kernel_size=9)])
        ...     dphysics = distribute(physics, ctx)

        Distribute a DataFidelity object:

        >>> from deepinv.optim.data_fidelity import L2
        >>> from deepinv.distributed import DistributedContext, distribute
        >>> with DistributedContext() as ctx: # doctest: +SKIP
        ...     data_fidelity = L2()
        ...     ddata_fidelity = distribute(data_fidelity, ctx)

        Distribute a Prior object:

        >>> from deepinv.models import DnCNN
        >>> from deepinv.distributed import DistributedContext, distribute
        >>> with DistributedContext() as ctx: # doctest: +SKIP
        ...     denoiser = DnCNN()
        ...     signal_shape = (1, 3, 256, 256)
        ...     ddenoiser = distribute(denoiser, ctx, signal_shape=signal_shape)
    """
    # Check object type and distribute accordingly
    if type_object == "auto":
        if isinstance(object, (StackedPhysics, StackedLinearPhysics)) or (
            isinstance(object, list)
            and len(object) > 0
            and isinstance(object[0], Physics)
        ):
            type_object = (
                "linear_physics"
                if isinstance(object, (StackedLinearPhysics, list))
                and (
                    not isinstance(object, list) or isinstance(object[0], LinearPhysics)
                )
                else "physics"
            )
        elif isinstance(object, (DataFidelity, StackedPhysicsDataFidelity)) or (
            isinstance(object, list)
            and len(object) > 0
            and isinstance(object[0], DataFidelity)
        ):
            type_object = "data_fidelity"
        elif isinstance(object, Denoiser):
            type_object = "denoiser"
        elif callable(object):
            raise ValueError(
                "For callable objects, you must specify type_object parameter"
            )
        else:
            raise ValueError(f"Cannot auto-detect type for object: {type(object)}")

    if type_object in ["physics", "linear_physics"]:
        return _distribute_physics(
            object,
            ctx,
            dtype=dtype,
            num_operators=num_operators,
            type_object=type_object,
            gather_strategy=gather_strategy,
            **kwargs,
        )
    elif type_object == "denoiser":
        return _distribute_processor(
            object,
            ctx,
            dtype=dtype,
            gather_strategy=gather_strategy,
            tiling_strategy=tiling_strategy,
            patch_size=patch_size,
            receptive_field_size=receptive_field_size,
            tiling_dims=tiling_dims,
            max_batch_size=max_batch_size,
            **kwargs,
        )
    elif type_object == "data_fidelity":
        return _distribute_data_fidelity(
            object,
            ctx,
            num_operators=num_operators,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported type_object: {type_object}")
