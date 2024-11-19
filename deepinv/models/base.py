import torch


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
    def __init__(self, device='cpu'):
        super().__init__()
        self.to(device)

    def forward(self, x, sigma, **kwargs):
        r"""
        Applies denoiser :math:`\denoiser{x}`.

        :param torch.Tensor x: noisy input.
        :param torch.Tensor, float sigma: noise level.
        :returns: denoised tensor.
        """
        return NotImplementedError


class Reconstructor(torch.nn.Module):
    r"""
    Base class for reconstruction models.

    Provides a template for defining reconstruction models.

    The base class inherits from :class:`torch.nn.Module`.

    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.to(device)

    def forward(self, y, physics, **kwargs):
        r"""
        Applies denoiser :math:`\inversef{y, A}`.

        :param torch.Tensor y: measurements.
        :param deepinv.physics.Physics physics: forward model :math:`A`.
        :returns: denoised tensor.
        """
        return NotImplementedError
