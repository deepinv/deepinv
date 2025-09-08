import torch
import torch_dct as dct

from deepinv.physics.forward import Physics
import torch.nn.functional as F

modulo_floor = lambda x, t: x - t * torch.floor(x / t)
modulo_round = lambda x, t: x - t * torch.round(x / t)


class SpatialUnwrapping(Physics):
    r"""
    Spatial unwrapping forward operator.

    This class implements a forward operator for spatial unwrapping, where the input is wrapped modulo a threshold value.
    The operator can use either floor or round mode for the wrapping operation. It is useful for problems where the observed data is wrapped,
    such as in phase imaging, modulo imaging, or interferometry.

    The forward operator is defined as:

    .. math::

        y = W_t(x) = x - t \cdot \mathrm{q}(x / t)

    where :math:`W_t` is the wrapping operator, :math:`t` is the threshold, and :math:`\mathrm{q}` is either the rounding or flooring function depending on the mode.


    :param float threshold: The threshold value for the modulo operation (default: 1.0).
    :param str mode: Wrapping mode, either 'round' or 'floor' (default: 'round').
    :param kwargs: Additional arguments passed to the base Physics class.

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.physics.spatial_unwrapping import SpatialUnwrapping
        >>> x = torch.tensor([[0.5, 1.2, 2.7]])
        >>> physics = SpatialUnwrapping(threshold=1.0, mode="round")
        >>> y = physics(x)
        >>> print(y.round(1))
        tensor([[ 0.5,  0.2, -0.3]])

    """

    def __init__(self, threshold=1.0, mode="round", **kwargs):

        super().__init__(**kwargs)

        self.threshold = threshold
        self.mode = mode

        self.B = modulo_floor if mode == "floor" else modulo_round

    def forward(self, x, **kwargs):
        return self.sensor(self.A(self.noise(x, **kwargs), **kwargs))

    def A(self, x):
        return self.B(x, self.threshold)

    def finite_differences(self, x):
        # apply spatial finite differences
        Dh_x = F.pad(torch.diff(x, 1, dim=-1), (1, 0))
        Dv_x = F.pad(torch.diff(x, 1, dim=-2), (0, 0, 1, 0))
        return Dh_x, Dv_x

    def D(self, x):
        Dh_x, Dv_x = self.finite_differences(x)
        out = torch.stack((Dh_x, Dv_x), dim=-1)
        return out

    def WD(self, x):
        Dx = self.D(x)
        WDx = modulo_round(Dx, self.threshold)
        return WDx

    def A_dagger(self, y):
        phi = self.prox_l2(None, y, self.threshold)
        return phi

    def A_adjoint(self, y):

        Dh_y, Dv_y = self.finite_differences(y)
        Dh_y = modulo_round(Dh_y, self.threshold)
        Dv_y = modulo_round(Dv_y, self.threshold)

        rho = -(
            torch.diff(F.pad(Dh_y, (0, 1)), 1, dim=-1)
            + torch.diff(F.pad(Dv_y, (0, 0, 0, 1)), 1, dim=-2)
        )

        return rho

    def A_vjp(self, x, v):
        _, vjpfunc = torch.func.vjp(self.D, x)
        return vjpfunc(v)[0]

    def prox_l2(self, z, y, rho):
        r"""
        Compute the proximal operator of the :class:`deepinv.optim.ItohFidelity` term 
        using DCT with the close-form solution of :footcite:t:`ramirez2024phase` as follows

        .. math::
            \hat{x}_{i,j} = \texttt{DCT}^{-1}\left(
            \frac{\texttt{DCT}(D^{\top}W_t(Dy) + \frac{\rho}{2} z)_{i,j}}
            { \frac{\rho}{2} + 4 - (2\cos(\pi i / M) + 2\cos(\pi j / N))}
            \right)

        where :math:`D` is the finite difference operator and :math:`\texttt{DCT}` is the discrete cosine transform.
        """


        psi = self.A_adjoint(y)

        if z is not None:
            psi = psi + (rho / 2) * z

        NX, MX = psi.shape[-1], psi.shape[-2]
        I, J = torch.meshgrid(torch.arange(0, MX), torch.arange(0, NX), indexing="ij")
        I, J = I.to(psi.device), J.to(psi.device)

        I, J = I.unsqueeze(0).unsqueeze(0), J.unsqueeze(0).unsqueeze(0)

        if z is None:
            denom = 2 * (
                2 - (torch.cos(torch.pi * I / MX) + torch.cos(torch.pi * J / NX))
            )
        else:
            denom = 2 * (
                (rho / 4)
                + 2
                - (torch.cos(torch.pi * I / MX) + torch.cos(torch.pi * J / NX))
            )

        dct_psi = dct.dct_2d(psi, norm="ortho")

        denom = denom.to(psi.device)
        denom[..., 0, 0] = 1  # avoid division by zero

        dct_phi = dct_psi / denom

        phi = dct.idct_2d(dct_phi, norm="ortho")

        return phi
