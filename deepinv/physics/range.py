import torch
from deepinv.physics.forward import DecomposablePhysics


class Decolorize(DecomposablePhysics):
    r"""
    Converts RGB images to grayscale.

    Follows the `rec601 <https://en.wikipedia.org/wiki/Rec._601>`_ convention.

    Images must be tensors with 3 colour (RGB) channels, i.e. [*,3,*,*]
    The measurements are grayscale images.

    :param str device: device to use.
    |sep|

    :Examples:

        Decolorize a 3x3 image:

        >>> from deepinv.physics import Decolorize
        >>> x = torch.ones(1, 3, 3, 3) # Define constant 3x3 RGB image
        >>> physics = Decolorize()
        >>> physics(x)
        tensor([[[[1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000],
                  [1.0000, 1.0000, 1.0000]]]], grad_fn=<MulBackward0>)
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.mask = torch.nn.Parameter(torch.ones((1), device=device) * 0.66851)

    def V_adjoint(self, x):
        y = x[:, 0, :, :] * 0.4472 + x[:, 1, :, :] * 0.8781 + x[:, 2, :, :] * 0.1706
        return y.unsqueeze(1)

    def V(self, y):
        return torch.cat([y * 0.4472, y * 0.8781, y * 0.1706], dim=1)


# # test code
# if __name__ == "__main__":
#    device = "cuda:0"
#    import deepinv as dinv
#    device = "cpu"
#    x = torch.randn((1, 3, 32, 32), device=device)
#    physics = Decolorize(device=device)
#    y = physics(x)
#    print(physics.adjointness_test(x))
#    print(physics.compute_norm(x))
#    xhat = physics.A_adjoint(y)
#    dinv.utils.plot([x, xhat, y])
