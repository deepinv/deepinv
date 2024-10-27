from typing import Callable
import torch
from deepinv.physics.forward import Physics


class RenderingPhysics(Physics):
    r"""
    Forward rendering operator.

    The rendering operator renders (or synthesises a view) from a 2.5D or 3D representation
    into a 2D image, from a given camera pose specified by rotation and translation matrices.
    The pose may either be relative or absolute.
    This physics supports many methods e.g. Gaussian splatting.

    The rendering physics can be written as (where :math:`\Pi` is the camera pose):

    .. math::

        y=\text{render}(x,\Pi)

    The camera pose attributes are updated using the method ``update_parameters``
    to allow easy loading during training.

    :param Callable renderer: rendering function that takes inputs x, pose rot matrix and pose translation matrix.
    """

    def __init__(self, renderer: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.renderer = renderer

    def A(
        self, x: torch.Tensor, pose_R: torch.Tensor = None, pose_T: torch.Tensor = None
    ):
        """Rendering forward operator.

        :param torch.Tensor x: 2.5D or 3D object to be rendered.
        :param torch.Tensor pose_R: absolute or relative camera pose rotation matrix, where None is the identity
        :param torch.Tensor pose_T: absolute or relative camera pose translation matrix where None is zero
        """
        # Assume for now x of shape (1, ...)
        if (
            pose_R is not None
            and pose_T is not None
            and pose_R.shape[0] != pose_T.shape[0]
        ):
            raise ValueError("Pose parameters should have same batch dim.")

        ys = []
        for i in range(pose_R.shape[0]):
            ys.append(self.renderer(x, pose_R, pose_T))

        y = torch.cat(ys)

        # Kind of redundant
        self.update_parameters(pose_R=pose_R, pose_T=pose_T)

        return y

    def update_parameters(self, pose_R: torch.Tensor, pose_T: torch.Tensor, **kwargs):
        self.pose_R = torch.nn.Parameter(pose_R)
        self.pose_T = torch.nn.Parameter(pose_T)
