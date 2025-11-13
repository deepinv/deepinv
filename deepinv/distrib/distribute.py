r"""
Distributed Framework Factory API
=================================

This module provides simplified factory builders for the distributed framework,
keeping users in control of their objects while removing repetitive boilerplate code.

The factory API exposes configuration-driven builders that create distributed components:

**Main Components:**

- :class:`FactoryConfig`: Configuration for physics, measurements, and data fidelity
- :class:`TilingConfig`: Configuration for spatial tiling strategies
- :class:`DistributedBundle`: Container for all distributed objects
- :func:`make_distrib_core`: Builder function that creates all distributed components

**Key Benefits:**

- **Reduced Boilerplate**: No need to write factory functions manually
- **Configuration-Driven**: Easy to modify, reuse, and share configurations
- **Type Safety**: Configuration objects prevent common errors
- **Single Builder**: All distributed objects created with one function call

The returned objects are native DeepInverse distributed classes, so you retain
full control and can inspect internals, swap components, and customize behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List, Sequence, Tuple, Union

import torch

from deepinv.physics import Physics, LinearPhysics
from deepinv.physics.forward import StackedPhysics, StackedLinearPhysics
from deepinv.optim.data_fidelity import DataFidelity, L2
from deepinv.optim.prior import Prior
from deepinv.models.base import Denoiser

from deepinv.deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedPhysics,
    DistributedLinearPhysics,
    DistributedProcessing,
)

from deepinv.distrib.distribution_strategies.strategies import DistributedSignalStrategy


def distribute_physics(
    physics: Union[StackedPhysics, List[Physics], Callable[[int, torch.device, Optional[dict]], Physics]],
    ctx: DistributedContext,
    *,
    num_operators: Optional[int] = None,
    type_object: Optional[str] = "physics",
    dtype: Optional[torch.dtype] = torch.float32,
    gather_strategy: str = "concatenated",
    **kwargs
) -> Union[DistributedPhysics, DistributedLinearPhysics]:
    r"""
    Distribute a Physics object across multiple devices.

    :param Physics physics: Physics object to distribute
    :param DistributedContext ctx: distributed context manager
    :param None, torch.dtype dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
        Default is `'concatenated'`.
    :param kwargs: additional keyword arguments for DistributedPhysics

    :returns: Distributed version of the input Physics object
    :rtype: DistributedPhysics

    |sep|

    :Examples:

        Distribute a Physics object:

        >>> from deepinv.physics import Blur
        >>> physics = Blur(kernel_size=5)
        >>> ctx = DistributedContext(devices=["cuda:0", "cuda:1"])
        >>> dphysics = distribute_physics(physics, ctx)
    """
    # Physics factory
    if isinstance(physics, (StackedPhysics, StackedLinearPhysics)):
        # Extract physics_list from StackedPhysics
        physics_list_extracted = physics.physics_list
        num_operators = len(physics_list_extracted)

        def physics_factory(idx: int, device: torch.device, shared: Optional[dict]):
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

        def physics_factory(idx: int, device: torch.device, shared: Optional[dict]):
            return physics_list_extracted[idx].to(device)

    if type_object == "linear_physics":
        return DistributedLinearPhysics(
            ctx, num_ops=num_operators, factory=physics_factory, dtype=dtype, 
            gather_strategy=gather_strategy, **kwargs
        )
    else:
        return DistributedPhysics(
            ctx, num_ops=num_operators, factory=physics_factory, dtype=dtype,
            gather_strategy=gather_strategy, **kwargs
        )


def distribute_processor(
    processor: Union[Prior, Denoiser],
    ctx: DistributedContext,
    *,
    dtype: Optional[torch.dtype] = torch.float32,
    tiling_strategy: Optional[Union[str, DistributedSignalStrategy]] = None,
    patch_size: int = 256,
    receptive_field_size: int = 64,
    overlap: bool = False,
    max_batch_size: Optional[int] = None,
    gather_strategy: str = "concatenated",
    **kwargs
) -> DistributedProcessing:
    r"""
    Distribute a DeepInverse prior or denoiser across multiple devices.

    :param Union[Prior, Denoiser] processor: DeepInverse prior or denoiser to distribute
    :param DistributedContext ctx: distributed context manager
    :param None, torch.dtype dtype: data type for distributed object. Default is `torch.float32`.
    :param Optional[Union[str, DistributedSignalStrategy]] tiling_strategy: strategy for tiling the signal. Options are `'basic'`, `'smart_tiling'`, `'smart_tiling_3d'`, or a custom strategy instance. Default is `'smart_tiling'`.
    :param int patch_size: size of patches for tiling strategies. Default is 256.
    :param int receptive_field_size: receptive field size for overlap in tiling strategies. Default is 64.
    :param bool overlap: whether patches should overlap. Default is False.
    :param None, int max_batch_size: maximum number of patches to process in a single batch. If `None`, all patches are batched together. Set to 1 for sequential processing.
    :param str gather_strategy: strategy for gathering distributed results (currently unused for processors, kept for API consistency).
    :param kwargs: additional keyword arguments for DistributedProcessing

    :returns: Distributed version of the input processor
    :rtype: DistributedProcessing

    |sep|

    :Examples:

        Distribute a Prior object:

        >>> from deepinv.optim.prior import TV
        >>> prior = TV(weight=0.1)
        >>> ctx = DistributedContext(devices=["cuda:0", "cuda:1"])
        >>> dprior = distribute_processor(prior, ctx)

        Distribute a Denoiser object:

        >>> from deepinv.models import DnCNN
        >>> denoiser = DnCNN(channels=3, num_layers=17)
        >>> ctx = DistributedContext(devices=["cuda:0", "cuda:1"])
        >>> ddenoiser = distribute_processor(denoiser, ctx)
    """

    return DistributedProcessing(
        ctx=ctx,
        processor=processor,
        dtype=dtype,
        strategy=tiling_strategy,
        strategy_kwargs={
            "patch_size": patch_size,
            "receptive_field_size": receptive_field_size,
            "overlap": overlap,
        },
        max_batch_size=max_batch_size,
    )


def distribute(
    object: Union[
        StackedPhysics,
        List[Physics],
        Callable[[int, torch.device, Optional[dict]],Physics],
        Denoiser,
        Prior,
        Callable[[int, torch.device, Optional[dict]], Union[Prior, Denoiser]],
    ],
    ctx: DistributedContext,
    *,
    type_object: Optional[str] = "auto",
    num_operators: Optional[int] = None,
    dtype: Optional[torch.dtype] = torch.float32,
    gather_strategy: str = "concatenated",
    tiling_strategy: Optional[Union[str, DistributedSignalStrategy]] = None,
    patch_size: int = 256,
    receptive_field_size: int = 64,
    overlap: bool = False,
    max_batch_size: Optional[int] = None,
    **kwargs
) -> Union[DistributedPhysics, DistributedLinearPhysics, DistributedProcessing]:
    r"""
    Distribute a DeepInverse object across multiple devices.

    This function takes a DeepInverse object (Physics, DataFidelity, or Prior)
    and distributes it using the provided DistributedContext.

    :param Union[Physics, DataFidelity, Prior] object: DeepInverse object to distribute
    :param DistributedContext ctx: distributed context manager
    :param Optional[str] type_object: type of object to distribute. Options are `'physics'`, `'data_fidelity'`, `'prior'`, or `'auto'` for automatic detection. Default is `'auto'`.
    :param None, torch.dtype dtype: data type for distributed object. Default is `torch.float32`.
    :param str gather_strategy: strategy for gathering distributed results. Options are:
        - `'naive'`: Simple object serialization (best for small tensors)
        - `'concatenated'`: Single concatenated tensor (best for medium/large tensors, minimal communication)
        - `'broadcast'`: Per-operator broadcasts (best for heterogeneous sizes or streaming)
        Default is `'concatenated'`.
    :param kwargs: additional keyword arguments for specific distributed classes

    :returns: Distributed version of the input object
    :rtype: Union[DistributedLinearPhysics, DistributedDataFidelity, DistributedPrior]

    |sep|

    :Examples:

        Distribute a Physics object:

        >>> from deepinv.physics import Blur
        >>> physics = Blur(kernel_size=5)
        >>> ctx = DistributedContext(devices=["cuda:0", "cuda:1"])
        >>> dphysics = distribute(physics, ctx)

        Distribute a DataFidelity object:

        >>> from deepinv.optim.data_fidelity import L2
        >>> data_fidelity = L2()
        >>> ddata_fidelity = distribute(data_fidelity, ctx, physics=dphysics, measurements=dmeasurements)

        Distribute a Prior object:

        >>> from deepinv.optim.prior import TV
        >>> prior = TV(weight=0.1)
        >>> signal_shape = (1, 3, 256, 256)
        >>> dprior = distribute(prior, ctx, signal_shape=signal_shape)
    """
    # Check object type and distribute accordingly
    if type_object == "auto":
        if isinstance(object, (StackedPhysics, StackedLinearPhysics)) or (isinstance(object, list) and len(object) > 0 and isinstance(object[0], Physics)):
            type_object = "linear_physics" if isinstance(object, (StackedLinearPhysics, list)) and (not isinstance(object, list) or isinstance(object[0], LinearPhysics)) else "physics"
        elif isinstance(object, DataFidelity):
            type_object = "data_fidelity"
        elif isinstance(object, Prior):
            type_object = "prior"
        elif isinstance(object, Denoiser):
            type_object = "denoiser"
        elif callable(object):
            raise ValueError("For callable objects, you must specify type_object parameter")
        else:
            raise ValueError(f"Cannot auto-detect type for object: {type(object)}")

    if type_object in ["physics", "linear_physics"]:
        return distribute_physics(
            object,
            ctx,
            dtype=dtype,
            num_operators=num_operators, 
            type_object=type_object,
            gather_strategy=gather_strategy,
            **kwargs
        )
    elif type_object in ["prior", "denoiser"]:
        return distribute_processor(
            object,
            ctx,
            dtype=dtype,
            gather_strategy=gather_strategy,
            tiling_strategy=tiling_strategy,
            patch_size=patch_size,
            receptive_field_size=receptive_field_size,
            overlap=overlap,
            max_batch_size=max_batch_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported type_object: {type_object}")
