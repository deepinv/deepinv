from deepinv.physics.forward import LinearPhysics
from deepinv.physics import adjoint_function
import torch
import numpy as np
from deepinv.physics.functional import random_choice


class HyperSpectralUnmixing(LinearPhysics):
    r"""
    Hyperspectral Unmixing.

    Hyperspectral Unmixing (HU) is a process that analyzes data captured by a special type of camera called a hyperspectral sensor.
    Imagine a regular camera capturing light in three bands (red, green, blue) to form a color image. A hyperspectral sensor captures light across hundreds of narrow bands, providing a much richer spectral signature for each pixel in the image.

    Analogy: Unmixing Paint. Imagine a painting as a pixel in a hyperspectral image. The paint at that pixel is likely a mixture of various basic colors.
    Unmixing separates the overall color (spectrum) of the pixel into the amounts (abundances) of each base color (endmember) used to create the mixture.

    Please see the survey at https://core.ac.uk/download/pdf/12043173.pdf#page=4.52. for more details.

    We can model a hyperspectral image mathematically using a Linear Mixing Model (LMM). LMM assumes each pixel's spectrum (denoted by :math:`\mathbf{y}` )
    is a linear combination of the spectra of pure materials (endmembers) in the scene, represented by a matrix :math:`\mathbf{M}`,
    weighted by their fractional abundances in the pixel (represented by a vector :math:`\alpha`) as above where :math:`\epsilon` represents noise in the measurement :math:`\mathbf{y}`.

    HSU inverse problems aim to recover the endmember matrix M and the abundance vector :math:`\mathbf{\alpha}` for each pixel in the image, essentially separating the mixed signals.
    Then, the HU problem is defined as:

    .. math::

        \mathbf{Y}= \mathbf{M}\cdot\mathbf{\alpha} + \mathbf{\epsilon}

    where :math:`\mathbf{Y}` is a collected hyperspectral image,
    :math:`\mathbf{M}` is a matrix of endmember spectra,
    :math:`\mathbf{\alpha}` is a matrix of abundances


    The size of the endmember matrix :math:`\mathbf{M}` is :math:`(E,C)`, where :math:`E` is the number of endmembers (materials, e.g. water, rock, wood, etc.)
    and :math:`C` is the number of channels (bands) in the hyperspectral image.
    The hyperspectral image :math:`\mathbf{y}` is of size :math:`(B, C, H, W)`,
    where :math:`B` is the batch size, :math:`H` is the height and :math:`W` is the width of the image.
    The abundance vector :math:`\mathbf{\alpha}` is of size :math:`(B, E, H, W)`.
    By default, we set :math:`E=15`, :math:`C=64`, :math:`H=W=128`.

    :param float M: Matrix of endmembers.  Default: ``None``.
    :param float E: Number of endmembers.  Default: ``15``.
    :param float C: Number of bands. Default: ``64``.
    :param float H: Height of the image. Default: ``128``.
    :param float W: Width of the image. Default: ``64``.


    :Examples:

        HSU operator using defined mask, removing the second column of a 3x3 image:

        >>> from deepinv.physics import HyperSpectralUnmixing
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> E, C = 15, 64
        >>> B, H, W = 4, 128, 128
        >>> alpha = torch.rand((B, E, H, W))
        >>> y = physics.A(alpha)
        >>> print(physics.M.shape, alpha.shape, y.shape)
        torch.Size([15, 64]) torch.Size([4, 15, 128, 128]) torch.Size([4, 64, 128, 128])

    """

    def __init__(
        self, M=None, E=15, C=64, H=128, W=128, device=torch.device("cpu"), **kwargs
    ):
        super(HyperSpectralUnmixing, self).__init__()

        self.device = device
        if M is None:
            M = torch.rand((E, C), dtype=torch.float32)
            self.M = torch.nn.Parameter(M).to(device)
        else:
            self.M = M

        self.M = self.M.to(device)
        self.E = E
        self.C = C
        self.H = H
        self.W = W
        self.update_parameters(M=self.M, **kwargs)
        self.pinv = self.get_pinv()

    def A(self, Alpha, M=None, **kwargs):
        # r"""
        # Applies the endmembers matrix to the input abundances Alpha.
        #
        # :param torch.Tensor A: input abundances.
        # :param torch.Tensor M: endmembers matrix :math:`\mathbf{M}` to be applied to the input image.
        # """

        self.update_parameters(M=M, **kwargs)

        assert Alpha.shape[1:] == (self.E, self.H, self.W)

        return torch.einsum("ec,behw->bchw", self.M, Alpha)

    def get_pinv(self):
        return torch.linalg.pinv(self.M)

    def A_dagger(self, y):
        return torch.einsum("ce,bchw->behw", self.pinv, y)

    def A_adjoint(self, y, **kwargs):
        return torch.einsum("ce,bchw->behw", self.M.t(), y)

    def update_parameters(self, M=None, **kwargs):
        r"""
        Updates the current endmembers matrix.

        :param torch.Tensor M: New endmembers matrix to be applied to the input abundances.
        """
        if M is not None:
            self.M = torch.nn.Parameter(M.to(self.device), requires_grad=False)

        if hasattr(self.noise_model, "update_parameters"):
            self.noise_model.update_parameters(**kwargs)


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # physics = HyperSpectralUnmixing(device=device)
    #
    # E, C = 15, 64
    # B, H, W = 4, 128, 128
    #
    # a = torch.zeros((B, E, H, W))
    # a[...,0:120, 0:120]=1
    #
    # y = physics.A(a)
    #
    # x_adjoint = physics.A_adjoint(y)
    # x_dagger = physics.A_dagger(y)
    #
    # print(physics.M.shape, a.shape, y.shape, x_adjoint.shape, x_dagger.shape)
    #
    # print(torch.norm(x_dagger - a))
    # # torch.Size([15, 64]) torch.Size([4, 15, 128, 128]) torch.Size([4, 64, 128, 128])
    # import deepinv as dinv
    #
    # dinv.utils.plot(
    #     [
    #         y[0, 0:1, ...],
    #         y[0, 1:2, ...],
    #         y[0, 2:3, ...],
    #         y[0, 3:4, ...],
    #         y[0, 4:5, ...],
    #     ],
    #     ["C1", "C2", "C3", "C4", "C5"],
    # )

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Define image size and physics
    B, E, H, W = 4, 15, 128, 128
    C = 64
    img_size = (B, E, H, W)
    physics = HyperSpectralUnmixing(E=E, C=C, H=H, W=W, device=device)

    # Generate random abundance matrix
    x = torch.randn(img_size, device=device, dtype=dtype)

    # Compute r, y, and the error
    r = physics.A_adjoint(physics.A(x))
    y = physics.A(r)
    error = (physics.A_dagger(y) - r).flatten().mean().abs()
    print(f"Error: {error.item()}")  # Error: 3.477338239576966e-08
    assert error < 0.01
