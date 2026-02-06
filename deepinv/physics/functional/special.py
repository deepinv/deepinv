import torch


def native_hankel1(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Implements H1(n, x) = J(n, x) + i*Y(n, x) using native PyTorch.
    Only supports real-valued x as of PyTorch 2.6.

    :param int n: Order of the Hankel function.
    :param torch.Tensor x: Input tensor.
    :return: Complex tensor representing H1(n, x).
    """
    # Use torch.special functions for J and Y
    jn = torch.special.bessel_j(n, x)
    yn = torch.special.bessel_y(n, x)

    # Combine into a complex tensor: J + iY
    return torch.complex(jn, yn)


def hankel1(n: int, x: torch.Tensor) -> torch.Tensor:
    r"""
    Computes the Hankel function of the first kind :math:`H_1(n, x)` for order :math:`n` and input :math:`x`.

    Uses native Torch if available, otherwise falls back to SciPy.

    :param int n: Order of the Hankel function.
    :param torch.Tensor x: Input tensor.
    :return: Complex tensor representing :math:`H_1(n, x)`.
    """
    # 1. Check if we can use native Torch (requires version >= 1.9 for torch.special)
    # and specifically bessel_j/y which were stabilized in later 2.x versions.
    has_native_bessel = hasattr(torch.special, "bessel_j") and hasattr(
        torch.special, "bessel_y"
    )
    device = x.device

    # 2. Use native Torch if available and input is on GPU or requires grad
    if has_native_bessel and (x.is_cuda or x.requires_grad):
        try:
            return native_hankel1(n, x).to(device=device)
        except RuntimeError:
            # Fallback if torch.special fails for specific dtypes/complex inputs
            pass

    # 3. Fallback to SciPy (requires CPU transfer)
    # Transfer to CPU, convert to numpy, compute, then back to Torch
    try:
        import scipy.special
    except ImportError:
        raise ImportError(
            "SciPy or PyTorch version >= 2.6 is required for hankel1 computation."
        )
    out = scipy.special.hankel1(n, x.to("cpu"))

    return out.to(device=device)


def bessel_j(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the Bessel function of the first kind :math:`J_v(n, x)` for order :math:`n` and input :math:`x`.

    Uses native Torch if available, otherwise falls back to SciPy.

    :param int n: Order of the Bessel function.
    :param torch.Tensor x: Input tensor.
    :return: Tensor representing :math:`J_v(n, x)`.
    """
    # 1. Attempt Native PyTorch (Available in torch >= 1.9)
    device = x.device
    if hasattr(torch.special, "bessel_j"):
        try:
            # Note: bessel_j supports float/double and supports autograd
            return torch.special.bessel_j(n, x).to(device=device)
        except (TypeError, RuntimeError):
            # Fallback if the specific n (order) or x (dtype) is unsupported natively
            pass

    # 2. Fallback to SciPy
    try:
        import scipy.special
    except ImportError:
        raise ImportError(
            "SciPy or PyTorch version >= 2.6 is required for jv computation."
        )
    # We detach and move to CPU to avoid breaking the graph or moving the whole model
    out = scipy.special.jv(n, x.to("cpu"))

    # Return as a torch tensor on the original or requested device
    return out.to(device=device)
