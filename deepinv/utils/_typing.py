import torch 
from typing import Union


def _handle_sigma(
    sigma: Union[float, torch.Tensor],
    batch_size: int = None,
    ndim: int = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    *args,
    **kwarg,
):
    r"""
    Convert various noise level types to the appropriate format for batch processing.
        If `sigma` is a single float or int, the same value will be used for each sample in the batch.
        If `sigma` is a tensor, it should be of shape `(batch_size,)` or a scalar.
        If `sigma` is a list, it should be of length `batch_size` or `1`.

    It will reshape the tensor to `(batch_size, 1, ..., 1)` if `ndim` and `batch_size` are specified.

    :param float, torch.Tensor sigma: noise level.
    :param int batch_size: number of samples in the batch (optional).
    :param int ndim: number of dimensions of the input tensor (optional).
    :param torch.device device: device to which the tensor should be moved (optional).
    :param torch.dtype dtype: data type of the tensor (optional).
    :param args: additional positional arguments.
    :param kwarg: additional keyword arguments.

    :returns: noise levels for each sample in the batch adapted to the denoiser.
    """
    if isinstance(sigma, (float, int)):
        sigma = float(sigma)
    elif isinstance(sigma, torch.Tensor):
        sigma = sigma.squeeze().to(dtype=dtype, device=device)
    elif isinstance(sigma, list):
        sigma = torch.tensor(sigma, dtype=dtype, device=device).squeeze()
    elif isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma, dtype=dtype, device=device).squeeze()
    else:
        raise TypeError(
            f"Sigma must be a float, int, or torch.Tensor. Got {type(sigma)}."
        )

    # Will reshape to (batch_size,) if batch_size is not None
    if batch_size is not None:
        # duplicate sigma for each sample in the batch
        if isinstance(sigma, float):
            sigma = torch.tensor([sigma] * batch_size, dtype=dtype, device=device)
        elif sigma.ndim == 0:
            sigma = sigma.view(1).expand(batch_size)
        elif sigma.ndim == 1 and sigma.size(0) == 1:
            sigma = sigma.view(1).expand(batch_size)
        elif sigma.ndim == 1 and sigma.size(0) != batch_size:
            raise ValueError(
                f"Sigma tensor size {sigma.size(0)} does not match batch size {batch_size}."
            )

    # Will reshape to (batch_size, 1, ..., 1) if ndim is not None
    if ndim is not None:
        if isinstance(sigma, float):
            sigma = torch.tensor(sigma, dtype=dtype, device=device).view(
                1, *([1] * (ndim - 1))
            )
        elif sigma.ndim == 0:
            sigma = sigma.view(1, *([1] * (ndim - 1)))
        elif sigma.ndim == 1:
            sigma = sigma.view(-1, *([1] * (ndim - 1)))
        else:
            raise ValueError(
                f"Sigma tensor has {sigma.ndim} dimensions, expected 0 or 1."
            )
    return sigma