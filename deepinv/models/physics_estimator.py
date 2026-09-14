from __future__ import annotations
import torch
import torch.nn as nn


class PhysicsEstimator(nn.Module):
    r"""
    Base class for physics parameter estimators.

    Provides a template for defining estimators that predict physics parameters from input images.
    """

    def __init__(
        self,
    ):
        super(PhysicsEstimator, self).__init__()

    def forward(self, y: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        r"""
        Estimates physics parameters from the input image.

        :param torch.Tensor y: input image
        :return: (:class:`torch.Tensor` or :class:`dict`) estimated physics parameters
        """
        raise NotImplementedError("Subclasses must implement this method.")
