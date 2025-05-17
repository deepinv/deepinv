import torch
from deepinv.physics.forward import DecomposablePhysics


class Decolorize(DecomposablePhysics):
    r"""
    Converts n-channel images to grayscale.

    The image channels are multiplied by factors determined by the spectral response function (SRF), then summed to produce a grayscale image.

    We provide various ways of defining the SRF including the `rec601 <https://en.wikipedia.org/wiki/Rec._601>`_ convention for RGB images.

    In the adjoint operation, we multiply the grayscale image by the coefficients in the SRF.

    Images must be tensors with C channels, i.e. ``(B,C,H,W)``. The measurements are grayscale images.

    :param int channels: number of channels in the input image.
    :param str, tuple, list srf: spectral response function. Either pass in user-defined SRF (must be of length channels),
        or ``rec601`` (default) following the `rec601 <https://en.wikipedia.org/wiki/Rec._601>`_ convention,
        or ``flat`` for a flat SRF (i.e. averages channels), or ``random`` for random SRF (e.g. to initialise joint learning).
    :param str, torch.device device: device on which to perform the computations. Default: ``cpu``.

    |sep|

    :Examples:

        Decolorize a 3x3 image:

        >>> import torch
        >>> from deepinv.physics import Decolorize
        >>> x = torch.ones((1, 3, 3, 3), requires_grad=False) # 3x3 RGB image
        >>> physics = Decolorize()
        >>> physics(x)
        tensor([[[[1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000]]]])
    """

    def __init__(self, channels=3, srf="rec601", device="cpu", **kwargs):
        super().__init__(**kwargs)
        if srf is None or srf == "rec601":
            srf = torch.tensor([0.4472 * 0.66851, 0.8781 * 0.66851, 0.1706 * 0.66851])
        elif srf in ("average", "flat"):
            srf = torch.tensor([1 / channels] * channels)
        elif srf == "random":
            srf = torch.rand(channels, device=device)
            srf /= srf.sum()
        elif isinstance(srf, (tuple, list)):
            srf = torch.tensor(srf)
        else:
            raise ValueError("Invalid srf")

        if srf.size(0) < channels:
            # pad with zeros
            srf = torch.cat([srf, torch.tensor([0] * (channels - srf.size(0)))])

        elif srf.size(0) > channels:
            raise ValueError("srf should be of length equal to or less than channels.")

        assert torch.allclose(srf.sum(), torch.tensor(1.0), rtol=1e-4)

        self.register_buffer("srf", srf.to(device).view(1, channels, 1, 1))
        self.register_buffer(
            "mask", torch.linalg.vector_norm(self.srf, dim=1, keepdim=True)
        )
        self.to(device)

    def V_adjoint(self, x):
        if x.shape[1] != self.srf.shape[1]:
            raise ValueError("x should have same number of channels as SRF.")

        return torch.sum(x * self.srf / self.mask, dim=1, keepdim=True)

    def V(self, y):
        if y.shape[1] != 1:
            raise ValueError(
                "y should be grayscale i.e. have length 1 in the 1st dimension."
            )

        return (
            y.expand(y.shape[0], self.srf.shape[1], *y.shape[2:]) * self.srf / self.mask
        )
