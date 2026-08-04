r"""
Wiener Deconvolution model for image reconstruction.

Implements the Wiener filter as a :class:`deepinv.models.Reconstructor` subclass,
providing a closed-form frequency-domain solution for deconvolution and denoising
problems that use :class:`deepinv.physics.BlurFFT` or :class:`deepinv.physics.Denoising`
forward operators.
"""

from __future__ import annotations

import torch
from torch import Tensor

from deepinv.models.base import Reconstructor
from deepinv.physics.blur import (
    BlurFFT,
    _VALID_PRIORS,
    _is_zero_lambda,
    _lambda_to_gamma,
)
from deepinv.physics.forward import Denoising, Physics


class WienerDeconvolution(Reconstructor):
    r"""
    Wiener deconvolution reconstruction model.

    Computes a closed-form image reconstruction by evaluating the solution
    in the frequency domain using
    :meth:`~deepinv.physics.DecomposablePhysics.prox_l2` (with :math:`z = 0`).

    Depending on the choice of ``lambda_reg`` and ``prior``, the model solves
    one of the following regularised optimisation problems:

    **1. Flat Prior** (``lambda_reg`` is a scalar, ``prior="flat"`` or ``None``):

    Standard Tikhonov regularisation with a flat (frequency-independent)
    penalty on the signal:

    .. math::

        \hat{x} = \arg\min_x \; \frac{1}{2}\Vert Ax - y \Vert_2^2
                  + \frac{\lambda}{2}\Vert x \Vert_2^2

    **2. Laplacian Prior** (``lambda_reg`` is a scalar, ``prior="laplacian"``):

    Classical **Wiener-Hunt deconvolution**.  The 2-D discrete Laplacian
    operator :math:`L` penalises high frequencies to enforce spatial smoothness:

    .. math::

        \hat{x} = \arg\min_x \; \frac{1}{2}\Vert Ax - y \Vert_2^2
                  + \frac{\lambda}{2}\Vert L x \Vert_2^2

    where :math:`L` is defined by the :math:`3 \times 3` kernel:

    .. math::

        L = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}

    *(Note: This is not supported when using* :class:`~deepinv.physics.Denoising` *;
    see the error message for a workaround.)*

    **3. Custom Tensor** (``lambda_reg`` is a user-provided tensor):

    The tensor is used as a per-coefficient regularisation weight in the SVD
    basis of the forward operator, and ``prior`` is ignored.  The model solves:

    .. math::

        \hat{x} = \arg\min_x \; \frac{1}{2}\Vert Ax - y \Vert_2^2
                  + \frac{1}{2}\Vert \lambda^{1/2} \odot V^* x \Vert_2^2

    where :math:`V^*` maps into the SVD basis of :math:`A` (the operator applied
    by :meth:`~deepinv.physics.DecomposablePhysics.V_adjoint`) and :math:`\odot`
    is element-wise multiplication.

    For :class:`~deepinv.physics.BlurFFT` that basis is the Fourier basis
    (:math:`V^* = F`), so :math:`\lambda` is a **frequency-dependent**
    noise-to-signal PSD ratio :math:`\lambda(f) = S_n(f) / S_x(f)` and the
    solution is the classical Wiener filter:

    .. math::

        \hat{X}(f) = \frac{H^*(f)}{\vert H(f) \vert^2 + \lambda(f)} \, Y(f)

    .. warning::

        For :class:`~deepinv.physics.Denoising` the SVD basis is the identity
        (:math:`V^* = I`), so a tensor ``lambda_reg`` weights **pixels**, not
        frequencies, and must therefore be image-shaped: a half-spectrum tensor
        of width :math:`W/2 + 1` raises a broadcasting error.  Pass a scalar
        ``lambda_reg`` there, or — for a genuine frequency-domain weighting —
        use :class:`~deepinv.physics.BlurFFT` with a delta filter, which is
        denoising expressed as a deconvolution.

    .. note::

        This model requires the forward operator to be linear shift-invariant.
        Currently, only :class:`~deepinv.physics.BlurFFT` and
        :class:`~deepinv.physics.Denoising` are supported. Other
        :class:`~deepinv.physics.DecomposablePhysics` subclasses (e.g.
        :class:`~deepinv.physics.MRI`) are not supported, as frequency-varying
        regularisation requires the operator's SVD basis to be the Fourier basis.

    .. seealso::

        :meth:`deepinv.physics.BlurFFT.A_dagger` exposes the same reconstruction
        directly on the operator, as ``physics.A_dagger(y, wiener=True)``, for
        quick baselines that do not need a stand-alone model.

    :param float, torch.Tensor lambda_reg: Regularisation parameter :math:`\lambda`
        controlling the trade-off between data fidelity and regularisation.
        **Larger values produce smoother (more regularised) reconstructions.**

        - If a **scalar** ``float``, it is combined with ``prior``
          (as detailed above).  Setting ``lambda_reg=0`` returns the
          pseudo-inverse (no regularisation).
        - If a :class:`torch.Tensor`, it is a per-coefficient weight in the
          operator's SVD basis, and ``prior`` is ignored.  With
          :class:`~deepinv.physics.BlurFFT` that basis is the Fourier basis, so
          the tensor is a frequency-dependent NSR (noise-to-signal PSD ratio,
          :math:`\lambda(f) = S_n(f) / S_x(f)`); see the warning above for
          :class:`~deepinv.physics.Denoising`.  Entries equal to zero are
          clamped to a small positive value to avoid a division by zero.

    :param str, None prior: Regularisation prior.  Only used when ``lambda_reg``
        is a scalar.

        - ``None`` or ``"flat"``: Flat (constant) regularisation.
        - ``"laplacian"``: Uses the power spectrum of the 2-D discrete Laplacian
          to build a frequency-varying regularisation, penalising high
          frequencies more than low frequencies (Wiener-Hunt model).

    :param str, torch.device device: Device for the model.  Default: ``"cpu"``.

    |sep|

    :Examples:

        Wiener deconvolution of a blurred image with a flat prior:

        >>> import torch
        >>> from deepinv.physics import BlurFFT
        >>> from deepinv.models import WienerDeconvolution
        >>> x = torch.randn(1, 1, 8, 8)
        >>> filter = torch.ones(1, 1, 3, 3) / 9.0
        >>> physics = BlurFFT(img_size=(1, 8, 8), filter=filter)
        >>> model = WienerDeconvolution(lambda_reg=1.0)
        >>> with torch.no_grad():
        ...     x_hat = model(physics(x), physics)
        >>> x_hat.shape
        torch.Size([1, 1, 8, 8])

    """

    def __init__(
        self,
        lambda_reg: float | Tensor = 1.0,
        prior: str | None = None,
        device: str | torch.device = "cpu",
    ):
        r"""
        Instantiates the Wiener deconvolution reconstruction model.

        See the class docstring for detailed information.

        :param float, torch.Tensor lambda_reg: Regularisation parameter :math:`\lambda`.
        :param str, None prior: Regularisation prior (``None``, ``"flat"``, or ``"laplacian"``).
        :param str, torch.device device: Device for the model. Default: ``"cpu"``.
        """
        super().__init__(device=device)

        # --- Validate prior ---
        if prior not in _VALID_PRIORS:
            raise ValueError(
                f"Invalid prior '{prior}'.  Must be one of {_VALID_PRIORS}."
            )

        self.lambda_reg = lambda_reg
        self.prior = prior

    def forward(self, y: Tensor, physics: Physics, **kwargs) -> Tensor:
        r"""
        Reconstruct an image from measurements using Wiener deconvolution.

        :param torch.Tensor y: Measurement tensor.
        :param deepinv.physics.Physics physics: Forward physics operator.
            Must be an instance of :class:`~deepinv.physics.BlurFFT` or
            :class:`~deepinv.physics.Denoising`.
        :return: Reconstructed image tensor.
        :rtype: torch.Tensor
        :raises ValueError: If ``physics`` is not a supported type.
        """
        # --- Validate physics type ---
        if not isinstance(physics, (BlurFFT, Denoising)):
            raise ValueError(
                f"WienerDeconvolution requires physics to be an instance of "
                f"BlurFFT or Denoising, but got {type(physics).__name__}.  "
                f"Wiener deconvolution is only mathematically valid for "
                f"convolutional forward operators (and the identity/denoising "
                f"operator as a special case)."
            )

        # --- lambda_reg = 0 means no regularisation: the pseudo-inverse ---
        if _is_zero_lambda(self.lambda_reg):
            return physics.A_dagger(y)

        # --- Convert lambda_reg (regularisation weight) into the internal
        #     gamma (data fidelity weight) expected by prox_l2 ---
        gamma = _lambda_to_gamma(self.lambda_reg, self.prior, physics)

        # --- Compute the Wiener-filtered reconstruction ---
        # With z = 0, prox_l2 minimises the following, where gamma acts
        # element-wise in the SVD basis of A (V^* = F, the Fourier transform,
        # for BlurFFT; the identity for Denoising):
        #   argmin_x  1/2 ||Ax - y||^2  +  1/2 || gamma^{-1/2} \odot V^*x ||^2
        # Substituting gamma = 1/lambda_reg gives the objective documented in
        # the class docstring:
        #   argmin_x  1/2 ||Ax - y||^2  +  1/2 || lambda_reg^{1/2} \odot V^*x ||^2
        # The three cases are specialisations of that single objective:
        #   - scalar lambda_reg collapses to  lambda_reg/2 ||x||^2, since V is
        #     unitary and so ||V^*x|| = ||x||;
        #   - the Laplacian prior sets lambda_reg(f) = lambda_reg (|H_L(f)|^2 + eps),
        #     which is  lambda_reg/2 ||Lx||^2  by Parseval's theorem;
        #   - a tensor lambda_reg is used as-is, i.e. a frequency-dependent NSR.
        return physics.prox_l2(z=0, y=y, gamma=gamma)
