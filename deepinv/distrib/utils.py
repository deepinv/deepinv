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
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any, Union

import torch
import deepinv as dinv

from deepinv.physics import Physics
from deepinv.optim.data_fidelity import DataFidelity, L2
from deepinv.optim.prior import Prior, PnP
from deepinv.loss.metric import PSNR

from deepinv.distrib.distrib_framework import (
    DistributedContext,
    DistributedLinearPhysics,
    DistributedDataFidelity,
    DistributedMeasurements,
    DistributedSignal,
    DistributedPrior,
)


# ---------------------------
# Configs & Bundles (lightweight)
# ---------------------------

@dataclass
class TilingConfig:
    r"""
    Configuration for spatial tiling strategies in distributed processing.

    :param int patch_size: size of patches for tiling operations
    :param int receptive_field_radius: radius of receptive field for overlap calculations
    :param bool overlap: whether to use overlapping patches
    :param str strategy: tiling strategy name. Options are `'basic'` and `'smart_tiling'`.
    """
    patch_size: int = 256
    receptive_field_radius: int = 64
    overlap: bool = False
    strategy: str = "smart_tiling"


@dataclass  
class FactoryConfig:
    r"""
    Configuration for distributed component factories.

    Specifies how to create physics operators, measurements, and data fidelity terms
    for the distributed framework. Can use either pre-created lists or factory functions.

    :param Union[Sequence[Physics], Callable] physics: either a list of physics operators or a factory function
    :param Union[Sequence[torch.Tensor], Callable] measurements: either a list of measurement tensors or a factory function
    :param None, int num_operators: required when using factory functions instead of lists
    :param None, Union[DataFidelity, Callable] data_fidelity: data fidelity term or factory function. If `None`, defaults to L2.
    """
    physics: Union[Sequence[Physics], Callable]
    measurements: Union[Sequence[torch.Tensor], Callable]
    num_operators: Optional[int] = None  # required if physics is a factory
    data_fidelity: Optional[Union[DataFidelity, Callable]] = None


@dataclass
class DistributedBundle:
    r"""
    Container for distributed framework components.

    Holds all the distributed objects created by the factory builder, providing
    convenient access to distributed physics, measurements, data fidelity, signal, and prior.

    :param DistributedLinearPhysics physics: distributed physics operators
    :param DistributedMeasurements measurements: distributed measurements
    :param DistributedDataFidelity df: distributed data fidelity
    :param DistributedSignal signal: distributed signal (always created)
    :param None, DistributedPrior prior: optional distributed prior
    """
    physics: DistributedLinearPhysics
    measurements: DistributedMeasurements
    data_fidelity: DistributedDataFidelity
    signal: DistributedSignal
    prior: Optional[DistributedPrior] = None


# ---------------------------
# Public Builders
# ---------------------------

def make_distrib_bundle(
    ctx: DistributedContext,
    *,
    factory_config: FactoryConfig,
    signal_shape: Tuple[int, int, int, int],
    reduction: str = "mean",
    prior: Optional[Prior] = None,
    tiling: Optional[TilingConfig] = None,
) -> DistributedBundle:
    r"""
    Create distributed components using factory configuration.

    This builder function creates all distributed components in a single call,
    reducing boilerplate code and ensuring consistent configuration.

    :param DistributedContext ctx: distributed context manager
    :param FactoryConfig factory_config: configuration specifying how to build physics, measurements, and data_fidelity
    :param Tuple[int, int, int, int] signal_shape: shape (B,C,H,W) for DistributedSignal
    :param str reduction: reduction strategy for DistributedDataFidelity. Options are `'mean'` and `'sum'`.
    :param None, Prior prior: optional prior term for distributed processing
    :param None, TilingConfig tiling: optional tiling configuration for spatial processing strategies

    :returns: Bundle containing all distributed objects
    :rtype: DistributedBundle

    |sep|

    :Examples:

        Create distributed components with list-based configuration:

        >>> factory_config = FactoryConfig(
        ...     physics=physics_list,
        ...     measurements=measurements_list,
        ...     data_fidelity=L2()
        ... )
        >>> bundle = make_distrib_bundle(ctx, factory_config=factory_config, signal_shape=(1,3,256,256))

        Create with custom factory functions:

        >>> def physics_factory(idx, device, shared):
        ...     return create_blur_operator(idx, device)
        >>> factory_config = FactoryConfig(
        ...     physics=physics_factory,
        ...     measurements=measurements_factory,
        ...     num_operators=4
        ... )
        >>> bundle = make_distrib_bundle(ctx, factory_config=factory_config, signal_shape=(1,3,256,256))
    """
    # Physics factory
    if callable(factory_config.physics):
        physics_factory = factory_config.physics
        num_operators = getattr(factory_config, "num_operators", None)
        if num_operators is None:
            raise ValueError("When using a factory for physics, you must provide num_operators as an attribute of FactoryConfig.")
    else:
        physics_list = factory_config.physics
        num_operators = len(physics_list)
        def physics_factory(idx: int, device: torch.device, shared: Optional[dict]):
            return physics_list[idx].to(device)

    # Measurements factory
    if callable(factory_config.measurements):
        measurements_factory = factory_config.measurements
        num_operators = getattr(factory_config, "num_operators", None)
        if num_operators is None:
            raise ValueError("When using a factory for measurements, you must provide num_operators as an attribute of FactoryConfig.")
    else:
        measurements_list = factory_config.measurements
        num_operators = len(measurements_list)
        def measurements_factory(idx: int, device: torch.device, shared: Optional[dict]):
            return measurements_list[idx].to(device)

    # Data fidelity factory
    if factory_config.data_fidelity is None:
        def df_factory_none(idx: int, device: torch.device, shared: Optional[dict]):
            return L2()
        df_factory = df_factory_none
    elif callable(factory_config.data_fidelity) and not isinstance(factory_config.data_fidelity, DataFidelity):
        # This is a proper factory function, not a data fidelity instance
        df_factory = factory_config.data_fidelity
    else:
        # This is a data fidelity instance (like L2()) or other object - create wrapper factory
        df_instance = factory_config.data_fidelity
        def df_factory_instance(idx: int, device: torch.device, shared: Optional[dict]):
            return df_instance
        df_factory = df_factory_instance

    physics = DistributedLinearPhysics(ctx, num_ops=num_operators, factory=physics_factory)
    measurements = DistributedMeasurements(ctx, num_items=num_operators, factory=measurements_factory)
    data_fidelity = DistributedDataFidelity(ctx, physics, measurements, data_fidelity_factory=df_factory, reduction=reduction)

    signal = DistributedSignal(ctx, shape=signal_shape)
    dprior = None

    # Optional prior
    if prior is not None:

        if tiling is None:
            tiling = TilingConfig()

        dprior = DistributedPrior(
            ctx=ctx,
            prior=prior,
            strategy=tiling.strategy,
            signal_shape=signal_shape,
            strategy_kwargs={
                "patch_size": tiling.patch_size,
                "receptive_field_radius": tiling.receptive_field_radius,
                "overlap": tiling.overlap,
            },
        )

    return DistributedBundle(physics=physics, measurements=measurements, data_fidelity=data_fidelity, signal=signal, prior=dprior)
