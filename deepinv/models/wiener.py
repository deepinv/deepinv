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
from deepinv.physics.blur import BlurFFT
from deepinv.physics.forward import Denoising, Physics
from deepinv.physics.functional import filter_fft

# Valid string values for the ``prior`` parameter.
_VALID_PRIORS = (None, "flat", "laplacian")


def _build_laplacian_gamma(
    gamma: float,
    physics: Physics,
    eps: float = 1e-9,
) -> Tensor:
    r"""Build a frequency-varying gamma tensor from a scalar gamma and a Laplacian prior.

    The 2-D discrete Laplacian kernel is::

        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]]

    Its power spectrum :math:`|H_L(f)|^2` is larger at high frequencies, therefore
    using this operator as a regularisation term in the objective function penalises
    high frequencies more than low frequencies. This enforces spatial smoothness,
    which is the standard assumption for natural images (Wiener-Hunt deconvolution).

    The returned tensor satisfies
    :math:`\gamma(f) = \frac{\gamma_{\text{scalar}}}{|H_L(f)|^2 + \varepsilon}`
    so that ``physics.prox_l2(z=0, y=y, gamma=gamma_tensor)`` yields the
    Wiener-Hunt solution.

    :param float gamma: Scalar regularisation strength.
    :param deepinv.physics.Physics physics: The forward physics operator. Used to determine
        the spatial dimensions :math:`(H, W)` so that the returned tensor
        correctly broadcasts with ``physics.mask`` (which contains the singular values
        of the forward operator, such as the Fourier magnitude of the blur kernel) inside
        :meth:`~deepinv.physics.DecomposablePhysics.prox_l2`.
    :param float eps: Small constant added to :math:`|H_L(f)|^2` to avoid
        division by zero at DC (where the Laplacian is zero). Default is ``1e-9``.
    :return: Frequency-varying gamma tensor whose shape is compatible with
        ``physics.mask`` after the trailing-dimension expansion performed
        by :meth:`~deepinv.physics.DecomposablePhysics.prox_l2`.
    :rtype: torch.Tensor
    :raises ValueError: If ``physics`` is a :class:`~deepinv.physics.Denoising` operator,
        because image dimensions are unknown.
    """
    # --- Determine spatial dimensions from the physics operator ----------
    if isinstance(physics, BlurFFT):
        if getattr(physics, "mask", None) is None:
            raise ValueError(
                "The BlurFFT operator must have a filter initialized before "
                "constructing the Laplacian prior."
            )

        # BlurFFT stores img_size as (C, H, W).
        H, W = physics.img_size[-2], physics.img_size[-1]

        # Future-proofing:
        # handles both 4D and 5D real-pair tensor formats
        # (if switched away from view_as_real() in V_adjoint() in BlurFFT)
        # and half / full spectrum FFTs (if switched to fft2 from rfft2 in BlurFFT).
        # DeepInverse filters are 4D: (1, C, H, W).
        # Determine the frequency width (W_freq) by checking the mask's dimensions.
        # A spatial-frequency mask is 4D: (1, C, H, W_freq).
        # To broadcast the real singular values against the real-pair representation
        # of view_as_real outputs, a 5th dimension is added: (1, C, H, W_freq, 2).
        if physics.mask.dim() > 4:
            W_freq = physics.mask.shape[-2]  # Extract from 5D tensor
        else:
            W_freq = physics.mask.shape[-1]  # Extract from 4D tensor

        use_real_fft = (W_freq == (W // 2 + 1))
    elif isinstance(physics, Denoising):
        # Denoising uses spatial-domain identity operators for its SVD (V = I).
        # We cannot apply a frequency-domain Laplacian prior inside its prox_l2.
        raise ValueError(
            "Cannot use prior='laplacian' with Denoising physics because "
            "Denoising does not use frequency-domain SVD operators. "
            "To apply a Laplacian prior to a denoising problem, bypass this limitation "
            "by treating Denoising as a special case of Deconvolution: pass a delta "
            "function as the filter to BlurFFT, which provides the necessary "
            "frequency-domain SVD."
        )
    else:
        # Should not reach here due to the validation in forward(), but
        # guard defensively.
        raise TypeError(
            f"Unsupported physics type {type(physics).__name__} for "
            f"Laplacian prior construction."
        )

    # --- Build the 2-D discrete Laplacian kernel -------------------------
    # Shape: (1, 1, 3, 3) — standard 4-connected Laplacian.
    laplacian_kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=physics.mask.device,
        dtype=physics.mask.real.dtype,
    ).reshape(1, 1, 3, 3)

    # --- Compute the Laplacian's power spectrum --------------------------
    # Zero-pad to (H, W) and compute centred FFT, matching BlurFFT's
    # convention (real_fft=True → rfft2 → half-spectrum of width W//2+1).

    # filter_fft expects img_size with same ndim as the filter tensor.
    # Our kernel is (1, 1, 3, 3) so img_size must be (1, 1, H, W).
    laplacian_fft = filter_fft(
        laplacian_kernel,
        img_size=(1, 1, H, W),
        real_fft=use_real_fft,
        dims=(-2, -1),
    )
    # laplacian_fft is a complex tensor (though mathematically purely real
    # because the symmetric kernel was circularly shifted to the spatial origin
    # in filter_fft()).
    # Shape: (1, 1, H, W_freq) is determined by use_real_fft.
    # |H_L(f)|^2 is the power spectrum.
    laplacian_power = laplacian_fft.real**2 + laplacian_fft.imag**2
    # laplacian_power shape: (1, 1, H, W_freq) : 4D
    #
    # When BlurFFT's mask is 5D: (1, C, H, W_freq, 2) due to view_as_real(),
    # prox_l2's expansion logic adds (mask.dim() - gamma.dim()) = 1
    # trailing dimension to gamma_tensor, giving (1, 1, H, W_freq, 1), which
    # correctly broadcasts with the 5D mask.


    # --- Compute gamma(f) = gamma_scalar / (|H_L(f)|^2 + eps) -----------
    gamma_tensor = gamma / (laplacian_power + eps)

    return gamma_tensor


class WienerDeconvolution(Reconstructor):
    r"""
    Wiener deconvolution reconstruction model.

    Computes a closed-form image reconstruction by evaluating the solution
    in the frequency domain using
    :meth:`~deepinv.physics.DecomposablePhysics.prox_l2` (with :math:`z = 0`).

    The model solves the following general regularised optimisation problem:

    .. math::

        \hat{x} = \arg\min_x \; \frac{1}{2}\Vert Ax - y \Vert_2^2 + \frac{1}{2} \Vert \tilde{\gamma}^{-1/2} \odot F x \Vert_2^2

    where :math:`F` is the Fourier transform, :math:`\odot` is element-wise multiplication,
    and :math:`\tilde{\gamma}` is an effective frequency-varying regularisation tensor.
    When :math:`A` is a convolution, this is equivalent to the classical Wiener filter.

    Depending on the choice of the ``gamma`` and ``prior`` arguments, the effective
    tensor :math:`\tilde{\gamma}` is defined such that the objective specialises into
    one of three specific cases:

    **1. Flat Prior** (``gamma`` is a scalar, ``prior="flat"`` or ``None``):

    :math:`\tilde{\gamma}` is a constant scalar equal to :math:`\gamma`.
    The objective mathematically reduces to standard Tikhonov regularisation:

    .. math::

        \hat{x} = \arg\min_x \; \frac{\gamma}{2}\Vert Ax - y \Vert_2^2 + \frac{1}{2}\Vert x \Vert_2^2

    **2. Laplacian Prior** (``gamma`` is a scalar, ``prior="laplacian"``):

    The scalar ``gamma`` constructs a frequency-varying regularisation tensor
    :math:`\tilde{\gamma}(f) = \gamma / (|H_L(f)|^2 + \varepsilon)`. Here, :math:`H_L(f)`
    is the frequency response (Fourier transform) of the 2D discrete Laplacian operator
    :math:`L`. In the spatial domain, :math:`L` is defined by the :math:`3 \times 3` kernel:

    .. math::

        L = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}

    (The kernel is zero-padded to the image dimensions before applying the FFT, allowing
    pointwise division in the Fourier domain). This penalises high frequencies to enforce
    spatial smoothness. It corresponds to the classical **Wiener–Hunt deconvolution**,
    solving the objective:

    .. math::

        \hat{x} = \arg\min_x \; \frac{\gamma}{2}\Vert Ax - y \Vert_2^2 + \frac{1}{2}\Vert L x \Vert_2^2

    **3. Custom Tensor** (``gamma`` is a user-provided tensor):

    :math:`\tilde{\gamma}` is directly set to the provided tensor ``gamma``, evaluating the
    general problem as defined above.

    .. note::

        This model requires the forward operator to be linear shift-invariant.
        Currently, only :class:`~deepinv.physics.BlurFFT` and
        :class:`~deepinv.physics.Denoising` are supported. Other
        :class:`~deepinv.physics.DecomposablePhysics` subclasses (e.g.
        :class:`~deepinv.physics.MRI`) are not supported, as frequency-varying
        regularisation requires the operator's SVD basis to be the Fourier basis.

    :param float, torch.Tensor gamma: Regularisation parameter controlling the
        trade-off between data fidelity and regularisation.

        - If a **scalar** ``float``, it is combined with ``prior`` (as detailed below).
        - If a :class:`torch.Tensor`, it is passed directly to ``prox_l2``
          and ``prior`` is ignored.  This allows advanced users to supply a
          pre-computed PSD ratio (e.g. :math:`\gamma(f) = S_x(f) / S_n(f)`).

    :param str, None prior: Regularisation prior.  Only used when ``gamma``
        is a scalar.

        - ``None`` or ``"flat"``: Flat (constant) SNR assumption.  The scalar
          ``gamma`` is passed directly to ``prox_l2``.
        - ``"laplacian"``: Uses the power spectrum of the 2-D discrete Laplacian
          to build a frequency-varying ``gamma`` tensor, penalising high
          frequencies more than low frequencies (Wiener–Hunt model).
          Note: This is not supported when using :class:`~deepinv.physics.Denoising`.

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
        >>> model = WienerDeconvolution(gamma=1.0)
        >>> with torch.no_grad():
        ...     x_hat = model(physics(x), physics)
        >>> x_hat.shape
        torch.Size([1, 1, 8, 8])

    """

    def __init__(
        self,
        gamma: float | Tensor = 1.0,
        prior: str | None = None,
        device: str | torch.device = "cpu",
    ):
        r"""
        Instantiates the Wiener deconvolution reconstruction model.

        See the class docstring for detailed information.

        :param float, torch.Tensor gamma: Regularisation parameter controlling the
            trade-off between data fidelity and regularisation.
        :param str, None prior: Regularisation prior (``None``, ``"flat"``, or ``"laplacian"``).
        :param str, torch.device device: Device for the model. Default: ``"cpu"``.
        """
        super().__init__(device=device)

        # --- Validate prior ---
        if prior not in _VALID_PRIORS:
            raise ValueError(
                f"Invalid prior '{prior}'.  Must be one of {_VALID_PRIORS}."
            )

        self.gamma = gamma
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

        # --- Determine the gamma value to pass to prox_l2 ---
        gamma = self.gamma

        if isinstance(gamma, Tensor):
            # User-supplied frequency-varying tensor: pass through as-is.
            # The ``prior`` parameter is ignored in this case.
            pass
        elif self.prior in (None, "flat"):
            # Scalar gamma with flat prior: pass the scalar directly.
            pass
        elif self.prior == "laplacian":
            # Scalar gamma with Laplacian prior: build a frequency-varying
            # gamma tensor from the Laplacian's power spectrum.
            gamma = _build_laplacian_gamma(gamma, physics)

        # --- Compute the Wiener-filtered reconstruction ---
        # prox_l2(z=0, y, gamma) solves:
        #   argmin_x  gamma/2 ||Ax - y||^2  +  1/2 ||x||^2
        # which is the Wiener/Tikhonov solution.
        return physics.prox_l2(z=0, y=y, gamma=gamma)
