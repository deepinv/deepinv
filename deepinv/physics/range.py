import torch
from deepinv.physics.forward import DecomposablePhysics


class Decolorize(DecomposablePhysics):
    r"""
    Converts RGB images to grayscale.

    Follows the `rec601 <https://en.wikipedia.org/wiki/Rec._601>`_ convention.

    Signals must be tensors with 3 colour (RGB) channels, i.e. [*,3,*,*]
    The measurements are grayscale images.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = 1.0

    def V_adjoint(self, x):
        y = x[:, 0, :, :] * 0.2989 + x[:, 1, :, :] * 0.5870 + x[:, 2, :, :] * 0.1140
        return y.unsqueeze(1)

    def V(self, y):
        return torch.cat([y * 0.2989, y * 0.5870, y * 0.1140], dim=1)


# # test code
# if __name__ == "__main__":
#     device = "cuda:0"
#
#     import deepinv as dinv
#     import matplotlib.pyplot as plt
#     import torchvision
#
#     dinv.device = "cpu"
#
#     x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
#     x = x.unsqueeze(0).float().to(dinv.device) / 255
#     x = torchvision.transforms.Resize((128, 128))(x)
#
#     physics = Decolorize()
#
#     y = physics(x)
#
#     print(physics.adjointness_test(x))
#     print(physics.compute_norm(x))
#     xhat = physics.A_adjoint(y)
#
#     plot_results = False  # set to True to plot results
#
#     if plot_results:
#         dinv.utils.plot([x, xhat, y])
