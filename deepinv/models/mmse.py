from __future__ import annotations
import torch
import torch.utils.data as data
from deepinv.models import Denoiser
from tqdm import tqdm
import numpy as np


class MMSE(Denoiser):
    r"""
    Closed-form MMSE denoiser for a Dirac-mixture prior based on a given dataset of images :math:`x_k`.

    .. math::

        p(x) = \frac{1}{N} \sum_{k=1}^N \delta(x - x_k)

    Given a noisy observation :math:`y = x + \sigma n` with :math:`n \sim \mathcal{N}(0, I)`, the MMSE estimate is given by:

    .. math::

        \mathbb{E}[x | y] = \sum_{k=1}^N x_k w(x_k \vert y)  \quad \text{with} \quad w(x_k \vert y) = \mathrm{softmax}\left( \left(- \frac{1}{\sigma^2}\|y - x_m\|^2 \right)_{m = 1, \cdots, N} \right)_k.

    Here, :math:`w(x_k \vert y)` is the posterior weight of atom :math:`x_k` knowing the measurement :math:`y`.

    :param torch.utils.data.DataLoader | torch.Tensor dataloader: Pytorch dataloader or tensor containing the dataset to use as prior.
        If a tensor is provided, it is assumed to contain all the dataset in memory.
        If the dataset is small, using a tensor can significantly speed up computations.
    :param torch.device, str device: Device to perform computations on. Default to CPU.
    :param torch.dtype dtype: dtype to compute the estimates. Default to `torch.float32`.
        For large datasets, using `torch.float16` or `torch.bfloat16` can significantly speed up computations.
        In this case, the accumulation is performed in `torch.float32` to avoid numerical issues.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> from torchvision import datasets
        >>> import torchvision.transforms.v2 as v2
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> dataset = datasets.MNIST(
        ...        root=".",
        ...        train=False,
        ...        download=True,
        ...        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            )
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
        >>> # Since the MNIST dataset is small, we can also load it entirely in memory as a tensor
        >>> dataloader = torch.cat([data[0] for data in iter(dataloader)]).to(device)
        >>> x = dataloader[0:4]
        >>> denoiser = dinv.models.MMSE(dataloader=dataloader, device=device, dtype=torch.float32)
        >>> sigma = 0.1
        >>> x_noisy = x + sigma * torch.randn_like(x)
        >>> with torch.no_grad():
        ...     x_denoised = denoiser(x_noisy, sigma=sigma)


    """

    def __init__(
        self,
        dataloader: data.DataLoader | torch.Tensor = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.dataloader = dataloader

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.tensor | float,
        *args,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if x.device.type != self.device.type:
            raise ValueError(
                f"Input tensor device {x.device.type} does not match model device {self.device.type}."
            )

        sigma = self._handle_sigma(
            sigma, batch_size=x.size(0), device=self.device, dtype=self.dtype, ndim=2
        )

        return MMSEFunction.apply(
            x, sigma, self.dataloader, self.device, self.dtype, verbose
        )

    def to(self, device: torch.device | str = None, dtype: torch.dtype = None):
        r"""
        Move the model to a specified device and/or dtype.

        :param torch.device | str device: Device to move the model to.
        :param torch.dtype dtype: Dtype to move the model to.
        :return: The model on the specified device and/or dtype.
        :rtype: MMSE
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self


def _select_accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    r"""
    Select appropriate accumulator dtype to avoid numerical issues.
    The distance computations can be computed faster in half precision
    """
    if dtype in [torch.float16, torch.bfloat16, torch.half]:
        return torch.float32
    return dtype


# Analytical implementation of the MMSE denoiser w.r.t the empirical distribution and its gradient
class MMSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sigma, dataloader, device, dtype, verbose):
        ctx.dataloader = dataloader
        ctx.device = device
        ctx.dtype = dtype
        ctx.verbose = verbose

        two_sigma_squared = 2 * sigma.pow(2)
        shape = x.shape
        Bx = x.size(0)

        x_flat = x.to(dtype).view(Bx, -1)

        acc_dtype = _select_accumulator_dtype(dtype)

        if isinstance(dataloader, torch.Tensor):
            ctx.is_tensor_mode = True
            batch = dataloader.to(device=device, dtype=dtype)
            if batch.ndim > 2:
                batch = batch.view(batch.size(0), -1)

            dist2 = torch.cdist(x_flat, batch).pow(2).to(acc_dtype)
            logw = -dist2 / (two_sigma_squared + 1e-12)

            weights = torch.nn.functional.softmax(logw, dim=1)

            x_denoised = weights @ batch.to(acc_dtype)
            x_denoised = x_denoised.view(shape)

            ctx.save_for_backward(x, sigma, x_denoised, weights, dist2)
            return x_denoised

        ctx.is_tensor_mode = False
        numerator = torch.zeros(Bx, np.prod(shape[1:]), device=device, dtype=acc_dtype)
        denominator = torch.zeros(Bx, 1, device=device, dtype=acc_dtype)
        shift = torch.full((Bx, 1), -torch.inf, device=device, dtype=acc_dtype)

        pb = tqdm(dataloader, desc="MMSE Denoiser", disable=not verbose)
        for batch in pb:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device=device, dtype=dtype, non_blocking=True)
            Bd = batch.size(0)
            batch = batch.view(Bd, -1)

            dist2 = torch.cdist(x_flat, batch).pow(2).to(acc_dtype, non_blocking=True)
            logw = -dist2 / (two_sigma_squared + 1e-12)

            batch_shift = logw.max(dim=1, keepdim=True).values
            new_shift = torch.maximum(shift, batch_shift)
            rescale = torch.exp(shift - new_shift)
            numerator = numerator * rescale
            denominator = denominator * rescale

            weights = torch.exp(logw - new_shift)
            numerator = numerator + weights @ batch.to(acc_dtype, non_blocking=True)
            denominator = denominator + weights.sum(dim=1, keepdim=True)
            shift = new_shift

        x_denoised = numerator / denominator
        x_denoised = x_denoised.view(shape)

        ctx.save_for_backward(x, sigma, x_denoised, shift, denominator)

        return x_denoised

    @staticmethod
    def backward(ctx, grad_output):
        dataloader = ctx.dataloader
        device = ctx.device
        dtype = ctx.dtype
        verbose = ctx.verbose

        if grad_output is None:
            return None, None, None, None, None, None

        if ctx.is_tensor_mode:
            x, sigma, x_denoised, weights, dist2 = ctx.saved_tensors
            batch = dataloader.to(device=device, dtype=dtype)
            if batch.ndim > 2:
                batch = batch.view(batch.size(0), -1)
        else:
            x, sigma, x_denoised, shift, denominator = ctx.saved_tensors

        Bx = x.size(0)
        x_flat = x.to(dtype).view(Bx, -1)
        grad_output_flat = grad_output.view(Bx, -1)
        x_denoised_flat = x_denoised.view(Bx, -1)

        acc_dtype = _select_accumulator_dtype(dtype)

        num_grad_x = torch.zeros_like(x_flat, dtype=acc_dtype)
        num_grad_sigma_1 = torch.zeros(Bx, 1, device=device, dtype=acc_dtype)
        num_grad_sigma_2 = torch.zeros(Bx, 1, device=device, dtype=acc_dtype)

        two_sigma_squared = 2 * sigma.pow(2)

        if ctx.is_tensor_mode:
            P = grad_output_flat @ batch.t()
            num_grad_x = (weights * P).to(acc_dtype) @ batch.to(acc_dtype)
            num_grad_sigma_1 = (weights * P * dist2).sum(dim=1, keepdim=True)
            num_grad_sigma_2 = (weights * dist2).sum(dim=1, keepdim=True)

            term1 = num_grad_x
            v_dot_x_hat = (grad_output_flat * x_denoised_flat).sum(dim=1, keepdim=True)
            term2 = x_denoised_flat * v_dot_x_hat

            grad_x = (term1 - term2) / sigma.pow(2)
            grad_x = grad_x.view(x.shape)

            term_sigma_1 = num_grad_sigma_1
            term_sigma_2 = num_grad_sigma_2

            grad_sigma = (term_sigma_1 - v_dot_x_hat * term_sigma_2) / sigma.pow(3)

            return grad_x, grad_sigma, None, None, None, None

        pb = tqdm(dataloader, desc="MMSE Backward", disable=not verbose)
        for batch in pb:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device=device, dtype=dtype, non_blocking=True)
            Bd = batch.size(0)
            batch = batch.view(Bd, -1)

            dist2 = torch.cdist(x_flat, batch).pow(2).to(acc_dtype, non_blocking=True)
            logw = -dist2 / (two_sigma_squared + 1e-12)

            weights = torch.exp(logw - shift)

            P = grad_output_flat @ batch.t()

            num_grad_x += (weights * P).to(acc_dtype) @ batch.to(acc_dtype)
            num_grad_sigma_1 += (weights * P * dist2).sum(dim=1, keepdim=True)
            num_grad_sigma_2 += (weights * dist2).sum(dim=1, keepdim=True)

        term1 = num_grad_x / denominator
        v_dot_x_hat = (grad_output_flat * x_denoised_flat).sum(dim=1, keepdim=True)
        term2 = x_denoised_flat * v_dot_x_hat

        grad_x = (term1 - term2) / sigma.pow(2)
        grad_x = grad_x.view(x.shape)

        term_sigma_1 = num_grad_sigma_1 / denominator
        term_sigma_2 = num_grad_sigma_2 / denominator

        grad_sigma = (term_sigma_1 - v_dot_x_hat * term_sigma_2) / sigma.pow(3)

        return grad_x, grad_sigma, None, None, None, None
