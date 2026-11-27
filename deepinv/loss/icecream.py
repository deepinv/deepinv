from __future__ import annotations

import torch
from torch import Tensor

from deepinv.loss.loss import Loss
from deepinv.physics.functional.cryo_et import apply_fourier_mask, missing_wedge_mask
from deepinv.transform.cube import CubeSymmetry


class IcecreamLoss(Loss):
    r"""
    Icecream self-supervised loss for cryo-electron tomography.

    Loss of :footcite:t:`kishore2026icecream`, training a network to denoise tomograms and correct their missing wedge without ground truth.
    Measurements are pairs :math:`y = (y_1, y_2)` of sub-tomograms from two halves of the dose, :math:`y_i = Mx + n_i`, with :math:`M` the :class:`missing-wedge operator <deepinv.physics.MissingWedge>` and independent noises :math:`n_i`.
    They are stacked along channels, and the model (e.g. :class:`deepinv.models.Icecream`) gives :math:`\hat x_i = f(y_i)`. The loss is the sum of:

    * a Noise2Noise term in the measured wedge, for denoising:

      .. math::

          \mathcal{L}_{\mathrm{obs}} = \| W M (y_2 - \hat x_1) \|^2 + \| W M (y_1 - \hat x_2) \|^2,

    * an equivariance term, for filling the missing wedge. For a random :class:`cube symmetry <deepinv.transform.CubeSymmetry>` :math:`T`, the network recovers the transformed estimate of one half from the measurement of the transformed estimate of the other, in the transformed measured wedge :math:`T M T^{-1}`:

      .. math::

          \mathcal{L}_{\mathrm{eq}} = \| W T M T^{-1} (B T \hat x_2 - f(M T \hat x_1)) \|^2 + \| W T M T^{-1} (B T \hat x_1 - f(M T \hat x_2)) \|^2,

      where :math:`B` restricts frequencies to the unit ball.

    The total loss is :math:`\mathcal{L}_{\mathrm{obs}} + \lambda \mathcal{L}_{\mathrm{eq}}`, where :math:`W` optionally keeps the central half of the sub-tomograms, unaffected by borders.
    Only symmetries with :math:`\|T M T^{-1} - M\| > \delta \|M\|` are used.

    :param float weight: weight :math:`\lambda` of the equivariance term.
    :param deepinv.transform.CubeSymmetry transform: transform sampling the symmetries. Default: all cube symmetries.
    :param float min_distance: minimum relative distance :math:`\delta` between the transformed and original wedges.
    :param bool window: if `True`, computes the loss on the central part of the sub-tomograms only.
    :param torch.nn.Module metric: metric comparing volumes. Default: MSE.

    |sep|

    :Example:

        >>> import torch
        >>> import deepinv as dinv
        >>> physics = dinv.physics.MissingWedge(img_size=(16, 16, 16), angle_range=(-60, 60))
        >>> model = dinv.models.Icecream(channels_per_scale=[4, 8])
        >>> loss = dinv.loss.IcecreamLoss()
        >>> y = torch.randn(2, 2, 16, 16, 16)  # pairs of sub-tomograms
        >>> x_net = model(y, physics)
        >>> loss(x_net=x_net, y=y, physics=physics, model=model).shape
        torch.Size([])
    """

    def __init__(
        self,
        weight: float = 2.0,
        transform: CubeSymmetry | None = None,
        min_distance: float = 0.5,
        window: bool = True,
        metric: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.weight = weight
        self.transform = CubeSymmetry() if transform is None else transform
        self.min_distance = min_distance
        self.window = window
        self.metric = torch.nn.MSELoss() if metric is None else metric
        self._allowed = None

    def allowed_symmetries(self, mask: Tensor) -> list[int]:
        r"""
        Symmetries of the transform moving the missing wedge by more than `min_distance`.

        :param torch.Tensor mask: missing-wedge mask on the unshifted Fourier grid.
        :return: (list[int]) symmetry indices.
        """
        key = (tuple(mask.shape), mask.device, float(mask.sum()))
        if self._allowed is None or self._allowed[0] != key:
            norm = mask.norm()
            allowed = [
                s
                for s in self.transform.symmetries
                if (CubeSymmetry.apply(mask, s, fourier=True) - mask).norm()
                > self.min_distance * norm
            ]
            if not allowed:
                raise ValueError(
                    "No symmetry moves the missing wedge enough, decrease min_distance."
                )
            self._allowed = (key, allowed)
        return self._allowed[1]

    def forward(self, x_net: Tensor, y: Tensor, physics, model, **kwargs) -> Tensor:
        r"""
        Computes the Icecream loss.

        :param torch.Tensor x_net: estimates :math:`(\hat x_1, \hat x_2)` of shape `(B, 2, N, N, N)`, i.e. `model(y, physics)`.
        :param torch.Tensor y: pairs of sub-tomograms :math:`(y_1, y_2)` of shape `(B, 2, N, N, N)`.
        :param deepinv.physics.MissingWedge physics: missing-wedge operator.
        :param torch.nn.Module model: network applied to each channel independently, e.g. :class:`deepinv.models.Icecream`.
        :return: (:class:`torch.Tensor`) loss.
        """
        if y.dim() != 5 or y.shape[1] != 2:
            raise ValueError(
                f"IcecreamLoss expects pairs of sub-tomograms of shape (B, 2, N, N, N), got {tuple(y.shape)}."
            )
        if len(set(y.shape[-3:])) != 1:
            raise ValueError("IcecreamLoss expects cubic sub-tomograms.")
        if x_net is None:
            x_net = model(y, physics)

        window = self._window(y)
        swap = [1, 0]  # cross pairing of the two halves
        loss_obs = self.metric(
            window * physics.A(y)[:, swap], window * physics.A(x_net)
        )

        allowed = self.allowed_symmetries(physics.mask)
        idx = torch.randint(
            len(allowed),
            (y.shape[0],),
            generator=self.transform.rng,
            device=self.transform.rng.device,
        )
        symmetries = [allowed[i] for i in idx.tolist()]
        x_rot = torch.cat(
            [CubeSymmetry.apply(x_net[[b]], s) for b, s in enumerate(symmetries)]
        )
        mask_rot = torch.stack(
            [CubeSymmetry.apply(physics.mask, s, fourier=True) for s in symmetries]
        )[:, None].to(y.dtype)
        ball = missing_wedge_mask(
            tuple(y.shape[-3:]), angle_range=(-90.0, 90.0), device=y.device
        ).to(y.dtype)

        reference = apply_fourier_mask(x_rot, ball)[:, swap]
        estimate = model(physics.A(x_rot), physics)
        loss_eq = self.metric(
            window * apply_fourier_mask(reference, mask_rot),
            window * apply_fourier_mask(estimate, mask_rot),
        )
        return loss_obs + self.weight * loss_eq

    def _window(self, y: Tensor) -> Tensor | float:
        if not self.window:
            return 1.0
        n = y.shape[-1]
        pad = n // 4
        window = torch.zeros(y.shape[-3:], device=y.device, dtype=y.dtype)
        window[pad : n - pad, pad : n - pad, pad : n - pad] = 1
        return window
