from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from copy import deepcopy

from torch.utils.data import DataLoader
from torch import Tensor
import torch

if TYPE_CHECKING:
    from deepinv.physics.generator.base import PhysicsGenerator
    from deepinv.physics.forward import Physics


class MultiOperatorMixin:
    """Mixin for multi-operator loss functions.

    Pass in factory args for a physics generator or a dataloader to return new physics params or new data samples.

    :param Callable physics_generator_factory: callable that returns a physics generator that returns new physics parameters
    :param Callable dataloader_factory: callable that returns a dataloader that returns new samples
    """

    def __init__(
        self,
        physics_generator_factory: Callable[..., PhysicsGenerator] = None,
        dataloader_factory: Callable[..., DataLoader] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.physics_generator = None
        self.dataloader = None

        if physics_generator_factory is not None:
            self.physics_generator = physics_generator_factory()

        if dataloader_factory is not None:
            self.dataloader = dataloader_factory()
            self.prev_epoch = -1
            self.reset_iter(epoch=0)

    def next_physics(self, physics: Physics, batch_size=1) -> Physics:
        """Return physics with new physics params.

        :param deepinv.physics.Physics physics: old physics.
        :param int batch_size: batch size, defaults to 1
        :return deepinv.physics.Physics: new physics.
        """
        if self.physics_generator is not None:
            physics_cur = deepcopy(physics)
            params = self.physics_generator.step(batch_size=batch_size)
            physics_cur.update(**params)
            return physics_cur
        return physics

    def physics_like(self, physics: Physics) -> Physics:
        """Copy physics, setting masks to ones (fully-sampled).

        :param deepinv.physics.Physics physics: input physics.
        :return deepinv.physics.Physics: new physics with fully-sampled mask of ones.
        """
        physics_new = deepcopy(physics)
        if hasattr(physics, "mask"):
            if isinstance(physics.mask, Tensor):
                physics_new.update(mask=torch.ones_like(physics.mask))
            elif isinstance(physics.mask, float):
                physics_new.update(mask=1.0)
            else:
                raise ValueError("physics mask must be either Tensor or float.")
        return physics_new

    def next_data(self) -> Tensor:
        """Return new data samples.
        :return torch.Tensor: new data samples.
        """
        if self.dataloader is not None:
            return next(self.iterator)

    def reset_iter(self, epoch: int) -> None:
        """Reset data iterator every epoch (to prevent `StopIteration`).
        :param int epoch: Epoch.
        """
        if epoch == self.prev_epoch + 1:
            self.iterator = iter(self.dataloader)
            self.prev_epoch += 1
