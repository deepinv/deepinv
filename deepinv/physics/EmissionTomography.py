import deepinv as dinv
import torch


class Emission_Tomography(dinv.physics.LinearPhysics):
    r"""
    Emission Tomography forward operator :math:`A` using the `pytomography` package (see : :footcite:t:`polson2025pytomography`).

    The forward problem is given by

    .. math::
        y ~ P(Ax + s)


    where :math:`P` is a Poisson noise model, :math:`A` is the system matrix modeling the physics of the imaging system,
    The operator :math:`A` can includes attenuation and point spread function (PSF) effects, :math:`x` is the
    image to reconstruct, :math:`s` is an optional additive scatter term and :math:`y` are the measured projections.

    :math:`x` is of shape (B, C, D, H, W) where B is the batch size, C is the number of channels (C=1 for emission tomography),
    D is the depth, H is the height and W is the width of the 3D image.
    :math:`y` is of shape (B, C, N_proj, H', W') where B is the batch size, C is the number of channels (C=1 for emission tomography),
    N_proj is the number of projections, H' is the height and W' is the width of the projections.

    :param object_meta: Metadata of the object (see `pytomography` documentation for more informations).
    :param proj_meta: Metadata of the projections (see `pytomography` documentation for more informations).
    :param att_transform: Attenuation transform (see `pytomography` documentation for more informations).
    :param psf_transform: Point Spread Function (PSF) transform (see `pytomography` documentation for more informations).
    """

    def __init__(self, object_meta, proj_meta, att_transform, psf_transform, **kwargs):
        super().__init__(**kwargs)
        from pytomography.projectors.SPECT import SPECTSystemMatrix

        att_transform.configure(object_meta, proj_meta)

        system_matrix = SPECTSystemMatrix(
            obj2obj_transforms=[att_transform, psf_transform],
            proj2proj_transforms=[],
            object_meta=object_meta,
            proj_meta=proj_meta,
        )

        self.system_matrix = system_matrix

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward operator.

        .. note::
            Pytomography does not support batch dimension, so we loop over the batch dimension.
            Channel dimension is squeezed since it should always be equal to 1, Pytomography
            only supports input of shape (D, H, W).

        :param torch.Tensor x: input tensor of shape (B, C, D, H, W).
        :returns: torch.Tensor y: output tensor of shape (B, C, N_proj, H', W').
        """
        x = x.squeeze(1)
        for batch in range(x.shape[0]):
            y_batch = self.system_matrix.forward(x[batch])
            if batch == 0:
                y = y_batch.unsqueeze(0)
            else:
                y = torch.cat((y, y_batch.unsqueeze(0)), dim=0)
        return y.unsqueeze(1)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Closed-form adjoint operator.

        .. note::
            Pytomography does not support batch dimension, so we loop over the batch dimension.
            Channel dimension is squeezed since it should always be equal to 1, Pytomography
            only supports input of shape (D, H, W).

        :param torch.Tensor y: input tensor of shape (B, C, N_proj, H', W').
        :returns: torch.Tensor x: output tensor of shape (B, C, D, H, W).
        """
        y = y.squeeze(1)
        for batch in range(y.shape[0]):
            x_batch = self.system_matrix.backward(y[batch])
            if batch == 0:
                x = x_batch.unsqueeze(0)
            else:
                x = torch.cat((x, x_batch.unsqueeze(0)), dim=0)
        return x.unsqueeze(1)
