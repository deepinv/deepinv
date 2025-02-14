from math import sqrt

import torch

from deepinv.physics.forward import LinearPhysics, adjoint_function
from deepinv.physics.functional import random_choice


class HyperSpectralUnmixing(LinearPhysics):
    r"""
    Hyperspectral Unmixing operator.

    Hyperspectral Unmixing (HU) analyzes data captured by a hyperspectral sensor,
    which captures light acorss a high number of bands (vs. a regular camera which captures light in three bands (RGB)).
    As an analogy, imagine the problem of unmixing paint in a pixel. The paint at a pixel is likely a mixture of various basic colors.
    Unmixing separates the overall color (spectrum) of the pixel into the amounts (abundances) of each base color (endmember) used to create the mixture.

    Please see the survey `Hyperspectral Unmixing Overview: Geometrical, Statistical, and Sparse Regression-Based Approaches <https://core.ac.uk/download/pdf/12043173.pdf>`_ for more details.

    Hyperspectral mixing is modelled using a Linear Mixing Model (LMM).

    .. math::

        \mathbf{y}= \mathbf{M}\cdot\mathbf{x} + \mathbf{\epsilon}

    where :math:`\mathbf{y}` is the resulting image of shape `(B, C, H, W)`. LMM assumes each pixel :math:`\mathbf{y}_i`'s spectrum
    is a linear combination of the spectra of pure materials (endmembers) in the scene, represented by a matrix :math:`\mathbf{M}` of shape :math:`(E,C)`,
    weighted by their fractional abundances in the pixel :math:`x_i` of shape :math:`(B, E, H, W)` where :math:`\epsilon` represents measurement noise.

    The HU inverse problem aims to recover the abundance vector :math:`\mathbf{x}` for each pixel in the image, essentially separating the mixed signals.
    If the endmember matrix :math:`\mathbf{M}` is unknown, then this must be estimated too.

    :param torch.Tensor M: Matrix of endmembers of shape :math:`(E,C)`. Overrides ``E`` and ``C`` parameters.
        If ``None``, then a random normalised matrix is simulated from a uniform distribution. Default ``None``.
    :param int E: Number of endmembers (e.g. number of materials). Ignored if ``M`` is set.  Default: ``15``.
    :param int C: Number of hyperspectral bands. Ignored if ``M`` is set. Default: ``64``.
    :param torch.device, str device: torch device, cpu or gpu.

    |sep|

    :Examples:

        Hyperspectral mixing of a 128x128 image with 64 channels and 15 endmembers:

        >>> from deepinv.physics import HyperSpectralUnmixing
        >>> E, C = 15, 64 # n. endmembers and n. channels
        >>> B, H, W = 4, 128, 128 # batch size and image size
        >>> physics = HyperSpectralUnmixing(E=E, C=C)
        >>> x = torch.rand((B, E, H, W)) # sample set of abundances
        >>> y = physics(x) # resulting mixed image
        >>> print(x.shape, y.shape, physics.M.shape)
        torch.Size([4, 15, 128, 128]) torch.Size([4, 64, 128, 128]) torch.Size([15, 64])

    """

    def __init__(
        self, M: torch.Tensor = None, E: int = 15, C: int = 64, device="cpu", **kwargs
    ):
        super(HyperSpectralUnmixing, self).__init__()
        self.device = device

        if M is None:
            # Simulate random normalised M
            M = torch.rand((E, C), dtype=torch.float32)
            M /= M.sum(dim=0, keepdim=True) * sqrt(C / E)

        self.E, self.C = M.shape

        self.update_parameters(M=M, **kwargs)
        self.M_pinv = torch.linalg.pinv(self.M)

    def A(self, x: torch.Tensor, M: torch.Tensor = None, **kwargs):
        r"""
        Applies the endmembers matrix to the input abundances x.

        :param torch.Tensor x: input abundances.
        :param torch.Tensor M: optional new endmembers matrix :math:`\mathbf{M}` to be applied to the input abundances.
        """
        self.update_parameters(M=M, **kwargs)

        if x.shape[1] != self.E:
            raise ValueError("Number of endmembers in x should be as defined.")

        return torch.einsum("ec,behw->bchw", self.M, x)

    def A_dagger(self, y: torch.Tensor, M: torch.Tensor = None, **kwargs):
        r"""
        Applies the pseudoinverse endmember matrix to the image y.

        :param torch.Tensor y: input image.
        :param torch.Tensor M: optional new endmembers matrix :math:`\mathbf{M}` to be applied to the input image.
        """
        self.update_parameters(M=M, **kwargs)

        if y.shape[1] != self.C:
            raise ValueError("Number of channels in y should be as defined.")

        return torch.einsum("ce,bchw->behw", self.M_pinv, y)

    def A_adjoint(self, y: torch.Tensor, M: torch.Tensor = None, **kwargs):
        r"""
        Applies the transpose endmember matrix to the image y.

        :param torch.Tensor y: input image.
        :param torch.Tensor M: optional new endmembers matrix :math:`\mathbf{M}` to be applied to the input image.
        """
        self.update_parameters(M=M, **kwargs)

        if y.shape[1] != self.C:
            raise ValueError("Number of channels in y should be as defined.")

        return torch.einsum("ce,bchw->behw", self.M.t(), y)

    def update_parameters(self, M: torch.Tensor = None, **kwargs):
        r"""
        Updates the current endmembers matrix.

        :param torch.Tensor M: New endmembers matrix to be applied to the input abundances.
        """
        if M is not None:
            if M.shape != (self.E, self.C):
                raise ValueError(
                    "Number of endmembers and bands should be same as before."
                )
            self.M = torch.nn.Parameter(M.to(self.device), requires_grad=False)

        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)
