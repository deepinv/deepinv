import torch
from typing import Union


class Denoiser(torch.nn.Module):
    r"""
    Base class for denoiser models.

    Provides a template for defining denoiser models.

    While most denoisers :math:`\denoisername` are designed to handle Gaussian noise
    with variance :math:`\sigma^2`, this is not mandatory.

    .. note::

        A Denoiser can be converted into a :class:`Reconstructor <deepinv.models.Reconstructor>`
        by using the :class:`deepinv.models.ArtifactRemoval` class.

    The base class inherits from :class:`torch.nn.Module`.

    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.to(device)

    def forward(self, x, sigma, **kwargs):
        r"""
        Applies denoiser :math:`\denoiser{x}{\sigma}`.

        :param torch.Tensor x: noisy input.
        :param torch.Tensor, float sigma: noise level.
        :returns: (:class:`torch.Tensor`) Denoised tensor.
        """
        return NotImplementedError

    @staticmethod
    def _handle_sigma(sigma: Union[float, torch.Tensor], *args, **kwarg):
        r"""
        Convert various noise level types to the appropriate format for batch processing.
            If `sigma` is a single float or int, the same value will be used for each sample in the batch.
            If `sigma` is a tensor, it should be of shape `(batch_size,)` or a scalar.

        To be overridden by subclasses if necessary.

        :param float, torch.Tensor sigma: noise level.
        :returns: list of noise levels for each sample in the batch.
        """
        if isinstance(sigma, (float, int)):
            return float(sigma)
        elif isinstance(sigma, torch.Tensor):
            return sigma.squeeze()
        else:
            raise TypeError(
                f"Sigma must be a float, int, or torch.Tensor. Got {type(sigma)}."
            )


class Reconstructor(torch.nn.Module):
    r"""
    Base class for reconstruction models.

    Provides a template for defining reconstruction models.

    Reconstructors provide a signal estimate ``x_hat`` as ``x_hat = model(y, physics)`` where ``y`` are the measurements
    and ``physics`` is the forward model :math:`A` (possibly including information about the noise distribution too).

    The base class inherits from :class:`torch.nn.Module`.

    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.to(device)

    def forward(self, y, physics, **kwargs):
        r"""
        Applies reconstruction model :math:`\inversef{y}{A}`.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward model :math:`A`.
        :returns: (:class:`torch.Tensor`) reconstructed tensor.
        """
        return NotImplementedError
