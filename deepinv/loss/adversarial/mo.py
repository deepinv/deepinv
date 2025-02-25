from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from copy import deepcopy

from torch.utils.data import DataLoader
from torch import Tensor

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
            physics_cur.update_parameters(**params)
            return physics_cur
        return physics

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
