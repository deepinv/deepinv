r"""
Wiener Deconvolution model for image reconstruction.

Implements the Wiener filter as a :class:`deepinv.models.Reconstructor` subclass,
providing a closed-form frequency-domain solution for deconvolution problems that
use a :class:`deepinv.physics.BlurFFT` forward operator.  Denoising is covered as
the special case of a unit-impulse filter.
"""

from __future__ import annotations

import torch
from torch import Tensor

from deepinv.models.base import Reconstructor
from deepinv.physics.blur import (
    BlurFFT,
    _WIENER_VALID_PRIORS,
    _is_zero_lambda,
    _lambda_to_gamma,
)
from deepinv.physics.forward import Denoising, Physics


class WienerDeconvolution(Reconstructor):
    r"""
    Wiener deconvolution reconstruction model.

    Solves the regularised least-squares problem

    .. math::

        \hat{x} = \arg\min_x \; \frac{1}{2}\Vert Ax - y \Vert_2^2 + \frac{1}{2}\Vert \lambda^{1/2} \odot F x \Vert_2^2

    where :math:`F` is the Fourier transform, :math:`\odot` is element-wise
    multiplication, and :math:`\lambda` is a per-frequency regularisation
    weight.  The minimiser is available in closed form and is computed directly
    with FFTs, without an iterative solver.

    The forward operator is a circular convolution, so
    :math:`A = F^{-1} \operatorname{diag}(H) F` is diagonalised by :math:`F`.
    The minimiser is therefore given by the classical Wiener filter,

    .. math::

        \hat{X}(f) = \frac{H^*(f)}{\vert H(f) \vert^2 + \lambda(f)} \, Y(f)

    in which :math:`\lambda(f)` plays the role of a noise-to-signal PSD ratio
    :math:`S_n(f) / S_x(f)`: it is small where the signal dominates, and large
    where the measurement is mostly noise.

    The arguments ``lambda_reg`` and ``prior`` define :math:`\lambda`,
    specialising the problem into one of three cases.

    **1. Flat Prior** (``lambda_reg`` is a scalar, ``prior="flat"`` or ``None``):

    :math:`\lambda(f) = \lambda` is constant.  Since :math:`F` is unitary, the
    penalty reduces to standard Tikhonov regularisation:

    .. math::

        \hat{x} = \arg\min_x \; \frac{1}{2}\Vert Ax - y \Vert_2^2 + \frac{\lambda}{2}\Vert x \Vert_2^2

    **2. Laplacian Prior** (``lambda_reg`` is a scalar, ``prior="laplacian"``):

    :math:`\lambda(f) = \lambda \left(|H_L(f)|^2 + \varepsilon\right)`, where
    :math:`H_L` is the frequency response of the 2-D discrete Laplacian
    :math:`L`.  Its power spectrum grows with frequency, so this penalises high
    frequencies more than low ones, enforcing spatial smoothness.  By
    Parseval's theorem this frequency-domain penalty is equivalent to a spatial
    penalty on :math:`Lx`, giving classical **Wiener-Hunt deconvolution**:

    .. math::

        \hat{x} = \arg\min_x \; \frac{1}{2}\Vert Ax - y \Vert_2^2 + \frac{\lambda}{2}\left(\Vert L x \Vert_2^2 + \varepsilon \Vert x \Vert_2^2\right)

    where :math:`L` is defined by the :math:`3 \times 3` kernel:

    .. math::

        L = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}

    The :math:`\varepsilon` term is numerically negligible
    (:math:`\varepsilon = 10^{-9}`), but it keeps the DC component regularised:
    the Laplacian has :math:`|H_L(0)|^2 = 0`, so without it the mean of the
    image would be left entirely unregularised.

    **3. Custom Tensor** (``lambda_reg`` is a user-provided tensor):

    :math:`\lambda(f)` is taken directly from the tensor, and ``prior`` is
    ignored.  This is the case to use when the noise and signal PSDs are known
    or have been estimated.

    .. note::

        Only :class:`~deepinv.physics.BlurFFT` is supported, since Wiener
        deconvolution requires a circular convolution.
        :class:`~deepinv.physics.Denoising` cannot express frequency-domain
        regularisation at all (see the tip below);
        :class:`~deepinv.physics.MRI` could, but subsamples the spectrum
        rather than blurring it, so the result would not be a deconvolution.

    .. tip::

        **Denoising** is the special case of deconvolution in which the filter
        is a unit impulse.  It is not supported through
        :class:`~deepinv.physics.Denoising`, whose SVD basis is the identity
        rather than the Fourier basis, and which therefore cannot express any
        frequency-domain prior.  Instead, build a
        :class:`~deepinv.physics.BlurFFT` with a filter that is zero except for
        a ``1`` at its centre::

            impulse = torch.zeros(1, 1, 3, 3)
            impulse[0, 0, 1, 1] = 1.0
            physics = BlurFFT(img_size=(C, H, W), filter=impulse)

        This gives :math:`A = I` together with a Fourier SVD basis, so
        ``prior="laplacian"`` and a frequency-dependent tensor ``lambda_reg``
        both apply.  Effective Wiener smoothing requires such a
        frequency-dependent weighting: a scalar :math:`\lambda` shrinks all
        frequencies uniformly.

    .. seealso::

        :meth:`deepinv.physics.BlurFFT.A_dagger` performs the same
        reconstruction directly on the operator, as
        ``physics.A_dagger(y, wiener=True)``.

    :param float, torch.Tensor lambda_reg: Regularisation parameter :math:`\lambda`
        controlling the trade-off between data fidelity and regularisation.
        **Larger values produce smoother (more regularised) reconstructions.**

        - If a **scalar** ``float``, it is combined with ``prior``
          (as detailed above).  Setting ``lambda_reg=0`` returns the
          pseudo-inverse (no regularisation).  A 0-dim tensor holds a single
          value and is treated the same way.
        - If a :class:`torch.Tensor` with one or more dimensions, it is a
          frequency-dependent NSR (noise-to-signal PSD ratio,
          :math:`\lambda(f) = S_n(f) / S_x(f)`), and ``prior`` is ignored.  It
          must be 4-D, of shape ``(B, C, H, W // 2 + 1)``, the half spectrum
          returned by :func:`torch.fft.rfft2`; ``B`` and ``C`` may be ``1`` to
          broadcast.  Entries equal to zero are clamped to a small positive
          value to avoid a division by zero.

    :param str, None prior: Regularisation prior.  Only used when ``lambda_reg``
        is a scalar.  Default: ``"laplacian"``.

        - ``"laplacian"``: Uses the power spectrum of the 2-D discrete Laplacian
          to build a frequency-varying regularisation, penalising high
          frequencies more than low frequencies (Wiener-Hunt model).
        - ``None`` or ``"flat"``: Flat (constant) regularisation.

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
        prior: str | None = "laplacian",
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
        if prior not in _WIENER_VALID_PRIORS:
            raise ValueError(
                f"Invalid prior '{prior}'.  Must be one of {_WIENER_VALID_PRIORS}."
            )

        # A tensor lambda_reg is registered as a buffer so that it moves with
        # the model on .to(device) and .cuda() calls.  It is non-persistent:
        # the model has no trainable parameters, and lambda_reg is a
        # user-supplied hyperparameter, so it is excluded from the state dict.
        if isinstance(lambda_reg, Tensor):
            self.register_buffer("lambda_reg", lambda_reg, persistent=False)
        else:
            self.lambda_reg = lambda_reg

        self.prior = prior

    def forward(self, y: Tensor, physics: Physics, **kwargs) -> Tensor:
        r"""
        Reconstruct an image from measurements using Wiener deconvolution.

        :param torch.Tensor y: Measurement tensor.
        :param deepinv.physics.Physics physics: Forward physics operator.
            Must be an instance of :class:`~deepinv.physics.BlurFFT`.  For
            denoising, use a :class:`~deepinv.physics.BlurFFT` built with a
            unit-impulse filter (see the class docstring).
        :param kwargs: Accepted for compatibility with the
            :class:`~deepinv.models.Reconstructor` interface and **ignored**.
            The reconstruction is fully determined by ``lambda_reg`` and
            ``prior``, which are set on the model at construction; passing
            them here has no effect.
        :return: Reconstructed image tensor.
        :rtype: torch.Tensor
        :raises ValueError: If ``physics`` is not a :class:`~deepinv.physics.BlurFFT`.
        """
        # --- Validate physics type ---
        # Only BlurFFT is accepted.  The notable rejected cases fail for
        # different reasons:
        #   - Denoising has the identity as its SVD basis, so no
        #     frequency-domain regularisation can be expressed.
        #   - Blur is convolutional but is a LinearPhysics, so it carries no
        #     SVD and prox_l2 there falls back to an iterative solve.  A
        #     closed-form Wiener filter needs BlurFFT.
        #   - MRI uses the Fourier basis, so a frequency-varying penalty is
        #     expressible, but it subsamples the spectrum rather than
        #     multiplying by a transfer function.  There is no H(f) to invert,
        #     so the result would be Tikhonov-regularised reconstruction rather
        #     than Wiener deconvolution.
        # Denoising is checked first so that its error message can point to the
        # unit-impulse alternative.
        if isinstance(physics, Denoising):
            raise ValueError(
                "WienerDeconvolution does not support Denoising physics: its SVD "
                "basis is the identity rather than the Fourier basis, so "
                "frequency-domain regularisation cannot be expressed.  Express "
                "denoising as a deconvolution instead, by passing a unit-impulse "
                "filter to BlurFFT: BlurFFT(img_size=(C, H, W), filter=impulse) "
                "where impulse is zero except for a 1 at its centre.  That also "
                "enables prior='laplacian' and frequency-dependent tensor "
                "lambda_reg, neither of which Denoising physics supports."
            )

        if not isinstance(physics, BlurFFT):
            raise ValueError(
                f"WienerDeconvolution requires physics to be an instance of "
                f"BlurFFT, but got {type(physics).__name__}.  The Wiener filter "
                f"is a closed-form spectral solution, so it needs an operator "
                f"that the Fourier transform diagonalises.  Being a convolution "
                f"is not sufficient: Blur is a LinearPhysics with no SVD, so "
                f"prox_l2 falls back to an iterative solve there.  For a "
                f"circular convolution, use BlurFFT(img_size=(C, H, W), "
                f"filter=filter) instead."
            )

        # --- lambda_reg = 0 means no regularisation: the pseudo-inverse ---
        if _is_zero_lambda(self.lambda_reg):
            return physics.A_dagger(y)

        # --- Convert lambda_reg (regularisation weight) into the gamma
        #     expected by prox_l2 ---
        gamma = _lambda_to_gamma(self.lambda_reg, self.prior, physics)

        # --- Compute the Wiener-filtered reconstruction ---
        # With z = 0, prox_l2 minimises the following, where gamma acts
        # element-wise in the SVD basis of A, which for BlurFFT is the Fourier
        # basis (V^* = F):
        #   argmin_x  1/2 ||Ax - y||^2  +  1/2 || gamma^{-1/2} \odot Fx ||^2
        # Substituting gamma = 1/lambda_reg gives the objective documented in
        # the class docstring:
        #   argmin_x  1/2 ||Ax - y||^2  +  1/2 || lambda_reg^{1/2} \odot Fx ||^2
        # The three cases are specialisations of that single objective:
        #   - scalar lambda_reg collapses to  lambda_reg/2 ||x||^2, since F is
        #     unitary and so ||Fx|| = ||x||;
        #   - the Laplacian prior sets lambda_reg(f) = lambda_reg (|H_L(f)|^2 + eps),
        #     which by Parseval's theorem is
        #     lambda_reg/2 (||Lx||^2 + eps ||x||^2);
        #   - a tensor lambda_reg is used as the frequency-dependent NSR, after
        #     clamping zero entries away from zero to avoid a division by zero.
        return physics.prox_l2(z=0, y=y, gamma=gamma)
