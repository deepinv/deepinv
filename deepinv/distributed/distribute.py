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
    DistributedStackedPhysics,
    DistributedStackedLinearPhysics,
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
) -> DistributedStackedPhysics | DistributedStackedLinearPhysics:
    r"""
    Distribute a Physics object across multiple devices.

    :param StackedPhysics | list[Physics] | Callable physics: Physics object to distribute.
        Can be a StackedPhysics, list of Physics objects, or a factory function.
    :param DistributedContext ctx: distributed context manager.
    :param int | None num_operators: number of physics operators when using a factory for physics, otherwise inferred. Default is `None`.
    :param str | None type_object: type of physics object to distribute. Options are `'physics'` or `'linear_physics'`. Default is `'physics'`.
    :param torch.dtype | None dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
        Default is `'concatenated'`.
    :param kwargs: additional keyword arguments for DistributedStackedPhysics.

    :return: Distributed version of the input Physics object.
    """
    # Physics factory
    if isinstance(physics, (StackedPhysics, StackedLinearPhysics)):
        # Extract physics_list from StackedPhysics
        physics_list_extracted = physics.physics_list
        num_operators = len(physics_list_extracted)

        def physics_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
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

        def physics_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
            return physics_list_extracted[idx].to(device)

    if type_object == "linear_physics":
        return DistributedStackedLinearPhysics(
            ctx,
            num_operators=num_operators,
            factory=physics_factory,
            dtype=dtype,
            gather_strategy=gather_strategy,
            **kwargs,
        )
    else:
        return DistributedStackedPhysics(
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
    tiling_strategy: str | DistributedSignalStrategy | None = None,
    patch_size: int = 256,
    overlap: int = 64,
    tiling_dims: int | tuple[int, ...] | None = None,
    max_batch_size: int | None = None,
    gather_strategy: str = "concatenated",
    **kwargs,
) -> DistributedProcessing:
    r"""
    Distribute a DeepInverse prior or denoiser across multiple devices.

    :param Prior | Denoiser processor: DeepInverse prior or denoiser to distribute.
    :param DistributedContext ctx: distributed context manager.
    :param torch.dtype | None dtype: data type for distributed object. Default is `torch.float32`.
    :param str | DistributedSignalStrategy | None tiling_strategy: strategy for tiling the signal. Options are `'basic'`, `'overlap_tiling'`, or a custom strategy instance. Default is `None` (which defaults to `'overlap_tiling'`).
    :param int patch_size: size of patches for tiling strategies. Default is `256`.
    :param int overlap: receptive field size for overlap in tiling strategies. Default is `64`.
    :param int | tuple[int, ...] | None tiling_dims: dimensions to tile over.
        Can be one of the following:
            - If ``None`` (default), tiles the last N-2 dimensions of your input tensor, i.e. for a tensor of shape (B, C, H, W), tiles over (H, W).
            - If an int ``N``, only tiles over the specified dimension.
            - If a tuple, specifies exact dimensions to tile.
    :param int | None max_batch_size: maximum number of patches to process in a single batch. If `None`, all patches are batched together. Set to `1` for sequential processing. Default is `None`.
    :param str gather_strategy: strategy for gathering distributed results (currently unused for processors, kept for API consistency). Default is `'concatenated'`.
    :param kwargs: additional keyword arguments for DistributedProcessing.

    :return: Distributed version of the input processor.
    """

    return DistributedProcessing(
        ctx=ctx,
        processor=processor,
        dtype=dtype,
        strategy=tiling_strategy,
        strategy_kwargs={
            "patch_size": patch_size,
            "overlap": overlap,
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

    :param DataFidelity | StackedPhysicsDataFidelity | list[DataFidelity] | Callable data_fidelity: DataFidelity object to distribute.
        Can be a DataFidelity, StackedPhysicsDataFidelity, list of DataFidelity objects, or a factory function.
    :param DistributedContext ctx: distributed context manager.
    :param int | None num_operators: number of data fidelity operators when using a factory for data_fidelity, otherwise inferred. Default is `None`.
    :param kwargs: additional keyword arguments for DistributedDataFidelity.

    :return: Distributed version of the input DataFidelity object.
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

        def data_fidelity_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
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

        def data_fidelity_factory(
            idx: int, device: torch.device, factory_kwargs: dict | None
        ):
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
    tiling_strategy: str | DistributedSignalStrategy | None = "overlap_tiling",
    tiling_dims: int | tuple[int, ...] | None = None,
    patch_size: int = 256,
    overlap: int = 64,
    max_batch_size: int | None = None,
    **kwargs,
) -> (
    DistributedStackedPhysics
    | DistributedStackedLinearPhysics
    | DistributedProcessing
    | DistributedDataFidelity
):
    r"""
    Distribute a DeepInverse object across multiple devices.

    This function takes a DeepInverse object and distributes it using the provided DistributedContext.

    The list of supported DeepInverse objects includes:

        - Physics operators: a list of :class:`deepinv.physics.Physics`, :class:`deepinv.physics.StackedPhysics` or :class:`deepinv.physics.StackedLinearPhysics`.
        - Data fidelity terms: a list of :class:`deepinv.optim.DataFidelity` or :class:`deepinv.optim.StackedPhysicsDataFidelity`.
        - Priors/Denoisers: :class:`deepinv.models.Denoiser` or :class:`deepinv.optim.Prior` objects.

    :param StackedPhysics | list[Physics] | Callable | Denoiser | DataFidelity | StackedPhysicsDataFidelity | list[DataFidelity] object: DeepInverse object to distribute. The supported types are listed above.
    :param DistributedContext ctx: distributed context manager.
    :param int | None num_operators: number of physics operators when using a factory for physics, otherwise inferred. Default is `None`.
    :param str | None type_object: type of object to distribute. Options are `'physics'`, `'linear_physics'`, `'data_fidelity'`, `'denoiser'`, or `'auto'` for automatic detection. Default is `'auto'`.
    :param torch.dtype | None dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results.

        Options are:
            - `'naive'`: Simple object serialization (best for small tensors)
            - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
            - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)

        Default is `'concatenated'`.

    :param str | DistributedSignalStrategy | None tiling_strategy: strategy for tiling the signal (for Denoiser).
        Options are `'basic'`, `'overlap_tiling'`, or a custom strategy instance. Default is `'overlap_tiling'`.
    :param int | tuple[int, ...] | None tiling_dims: dimensions to tile over (for Denoiser).

        Can be one of the following:
            - If ``None`` (default), tiles the last N-2 dimensions of your input tensor.
            - If an int ``N``, only tiles over the specified dimension.
            - If a tuple, specifies exact dimensions to tile.

        Examples:
            - For ``(B, C, H, W)`` image: ``tiling_dims=(2, 3)`` tiles over H and W.
            - For ``(B, C, D, H, W)`` volume: ``tiling_dims=(2, 3, 4)`` tiles over D, H, W.
            - For ``(B, C, H, W)`` image: ``tiling_dims=2`` tiles only over H dimension.
            - For ``(B, C, D, H, W)`` volume: ``tiling_dims=None`` tiles over D, H, W dimensions.

    :param int patch_size: size of patches for tiling strategies (for Denoiser).
        Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is `256`.
    :param int overlap: receptive field size for overlap in tiling strategies (for Denoiser).
        Can be an int (same size for all tiled dims) or a tuple (per-dimension size). Default is `64`.
    :param int | None max_batch_size: maximum number of patches to process in a single batch (for Denoiser). If `None`, all patches are batched together. Set to `1` for sequential processing. Default is `None`.
    :param kwargs: additional keyword arguments for specific distributed classes.

    :return: Distributed version of the input object.

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
        ...     ddenoiser = distribute(denoiser, ctx)
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
            overlap=overlap,
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
