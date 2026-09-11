"""Signal processing utilities"""

from __future__ import annotations
from warnings import warn

import torch


def normalize_signal(
    inp: torch.Tensor,
    *,
    mode: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> torch.Tensor:
    r"""
    Normalize a batch of signals between zero and one.

    :param torch.Tensor inp: the input signal to normalize, it should be of shape `(B, *)`.
    :param str mode: the normalization, either `'min_max'` for min-max normalization or `'clip'` for clipping.
        If ``clip`` is selected, the values of ``vmin`` and ``vmax`` are used as clipping bounds if provided,
        otherwise the default bounds of 0.0 and 1.0 are used.
        Note that min-max normalization of constant signals is ill-defined and here it amounts to mapping the constant
        value to the closest value between zero and one (which is equivalent to clipping).
    :return: the normalized batch of signals.

    """
    if mode != "clip":
        if vmin is not None or vmax is not None:
            warn(
                "The vmin and vmax arguments are used only when using 'clip' rescaling.",
                UserWarning,
                stacklevel=2,
            )
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError(
            f"vmin should be strictly less than vmax, got vmin={vmin} and vmax={vmax}."
        )
    if mode == "min_max":
        # Compute the minimum and maximum intensity of the batched signals
        non_batched_dims = list(range(1, inp.ndim))
        minimum_intensity = inp.amin(dim=non_batched_dims, keepdim=False)
        maximum_intensity = inp.amax(dim=non_batched_dims, keepdim=False)

        # Clone the signal to avoid input mutations
        inp = inp.clone()

        # The indices corresponding to the non-constant batched signals
        indices = maximum_intensity != minimum_intensity

        # Prepare the tensors for broadcasting
        shape = (-1,) + (1,) * len(non_batched_dims)
        minimum_intensity = minimum_intensity.view(*shape)
        maximum_intensity = maximum_intensity.view(*shape)

        # Rescale the non-constant batched signals between zero and one
        inp[indices] -= minimum_intensity[indices]
        inp[indices] /= maximum_intensity[indices] - minimum_intensity[indices]

        # The indices corresponding to the constant batched signals
        indices = torch.logical_not(indices)

        # Clamp constant batched signals between zero and one
        inp[indices] = inp[indices].clamp(min=0.0, max=1.0)
    elif mode == "clip":
        # Clamp every batched signal between zero and one
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0
        inp = inp.clamp(min=vmin, max=vmax)
        inp = (inp - vmin) / (vmax - vmin + 1e-12)  # rescale to [0, 1]

    else:  # pragma: no cover
        raise ValueError(
            f"Unsupported normalization mode: {mode}. Supported modes are 'min_max' and 'clip'."
        )

    return inp


def complex_abs(data: torch.Tensor | None, dim=1, keepdim=True):
    """
    Compute the absolute value of a complex valued input tensor.

    If data has length 2 in the channel dimension given by dim, assumes this represents Re and Im parts.
    If data is a ``torch.complex`` dtype, takes absolute directly.

    :param torch.Tensor data: A complex valued tensor.
    :param int dim: complex dimension
    :param bool keepdim: keep complex dimension after abs
    """
    if data is None:
        return data

    if data.is_complex():
        return torch.abs(data)
    else:
        if data.size(dim) != 2:  # pragma: no cover
            raise ValueError(
                f"Data must be torch.complex dtype or have length 2 in given dim, got data of type {data.dtype} and shape {tuple(data.shape)} which is not two at index {dim}"
            )
        return torch.linalg.vector_norm(data, dim=dim, ord=2, keepdim=keepdim)


def hilbert(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""
    Compute the analytical signal via Hilbert transform.

    .. note::
        This function uses ``scipy`` and is therefore not efficient nor differentiable. If a
        pure ``torch`` implementation is required, please raise a feature request issue on
        `GitHub <https://github.com/deepinv/deepinv/issues>`_.

    :param torch.Tensor x: real-valued input signal of arbitrary shape.
    :param int dim: dimension along which the transform is computed, e.g. the time axis of
        raw data or the depth axis of an image. (default: ``-1``)
    :return: (:class:`torch.Tensor`) the complex-valued analytical signal, of the same shape
        as ``x``.
    """
    from scipy.signal import hilbert

    if x.is_complex():
        raise ValueError(
            "The Hilbert transform expects a real-valued signal, "
            f"got dtype {x.dtype}. The analytical signal is already complex."
        )
    analytical = hilbert(x.detach().cpu().numpy(), axis=dim)
    dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    return torch.from_numpy(analytical).to(device=x.device, dtype=dtype)


def bmode(
    x: torch.Tensor,
    dim: int = -2,
    *,
    amplitude_floor_db: float = -60.0,
    dynamic_range: float | None = None,
    reference: float | torch.Tensor | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    r"""
    Compute log-compressed brightness mode (B-Mode) image.

    .. math::
        \mathrm{B}(x) = 20 \log_{10} \left(\frac{x_a}{x_\mathrm{ref}} \right),

    where :math:`x_a` is the envelope of :math:`x`, i.e. the modulus of its analytical signal
    (see :func:`deepinv.utils.hilbert`) or its modulus if :math:`x` is complex-valued, and
    :math:`x_\mathrm{ref}` a reference amplitude. The result is clipped to
    :math:`[\mathrm{amplitude\_floor\_db}, \mathrm{amplitude\_floor\_db} + \mathrm{dynamic\_range}]`.

    :param torch.Tensor x: input signal of shape ``(B, ...)``
    :param int dim: dimension along which the envelope is computed. (default: ``-2``)
    :param float amplitude_floor_db: lower bound of the display window, in dB relative to the reference. (default: ``-60``)
    :param float dynamic_range: width of the display window in dB. If ``None``, the window ends at 0 dB. (default: ``None``)
    :param float, torch.Tensor reference: reference amplitude mapped to 0 dB. If ``None``, the maximum of the envelope of each element of the batch. (default: ``None``)
    :param bool normalize: if ``True``, the display window is linearly mapped to ``[0, 1]``, which is convenient for display or for saving the image. (default: ``False``)
    :return: (:class:`torch.Tensor`) the log-compressed image, in dB or in ``[0, 1]`` if ``normalize`` is ``True``.
    """
    if dynamic_range is None:
        dynamic_range = -amplitude_floor_db
    if dynamic_range <= 0:
        raise ValueError(f"dynamic_range must be positive, got {dynamic_range}.")

    amplitude = x.abs() if x.is_complex() else hilbert(x, dim=dim).abs()

    if reference is None:
        dims = tuple(range(1, amplitude.ndim))
        reference = amplitude.amax(dim=dims, keepdim=True) if dims else amplitude.amax()
    reference = torch.as_tensor(
        reference, dtype=amplitude.dtype, device=amplitude.device
    ).clamp(min=torch.finfo(amplitude.dtype).tiny)

    ratio = (amplitude / reference).clamp(min=10.0 ** (amplitude_floor_db / 20.0))
    out = (20.0 * torch.log10(ratio)).clamp(max=amplitude_floor_db + dynamic_range)
    return (out - amplitude_floor_db) / dynamic_range if normalize else out
