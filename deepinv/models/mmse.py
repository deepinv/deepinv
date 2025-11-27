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

    :param dataloader: Pytorch dataloader containing the dataset to use as prior.
    :param device: Device to perform computations on. Default to CPU.
    :param dtype: dtype to compute the estimates. Default to `torch.float32`.
        For large datasets, using `torch.float16` or `torch.bfloat16` can significantly speed up computations.
        In this case, the accumulation is performed in `torch.float32` to avoid numerical issues.

    |sep|

    :Examples:

        >>> import deepinv as dinv
        >>> from torchvision import datasets
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> dataset = datasets.MNIST(root='.', train=False, download=True)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
        >>> x = next(iter(dataloader))[0].to(device)
        >>> denoiser = dinv.models.MMSE(dataloader=dataloader, device=device, dtype=torch.float32)
        >>> sigma = 0.1
        >>> x_noisy = x + sigma * torch.randn_like(x)
        >>> with torch.no_grad():
        ...     x_denoised = denoiser(x_noisy, sigma=sigma)

    """

    def __init__(
        self,
        dataloader: data.DataLoader = None,
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
        if x.device != self.device:
            raise ValueError(
                f"Input tensor device {x.device} does not match model device {self.device}."
            )

        sigma = self._handle_sigma(
            sigma, batch_size=x.size(0), device=self.device, dtype=self.dtype, ndim=2
        )
        two_sigma_squared = 2 * sigma.pow(2)  # (Bx,1)
        shape = x.shape
        Bx = x.size(0)

        x = x.to(self.dtype).view(Bx, -1)  # (Bx, 1, C*H*W)

        # Streaming accumulation across dataset atoms for each batch
        acc_dtype = self._select_accumulator_dtype(self.dtype)

        numerator = torch.zeros(
            Bx, np.prod(shape[1:]), device=self.device, dtype=acc_dtype
        )
        denominator = torch.zeros(Bx, 1, device=self.device, dtype=acc_dtype)
        # Row-wise shift for numerical stability in log-domain
        shift = torch.full((Bx, 1), -torch.inf, device=self.device, dtype=acc_dtype)

        pb = tqdm(self.dataloader, desc="MMSE Denoiser", disable=not verbose)
        for batch in pb:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )  # (Bd, C, H, W)
            Bd = batch.size(0)
            batch = batch.view(Bd, -1)  # (1, Bd, C*H*W)

            # Pairwise squared distances: (Bx, Bd)
            dist2 = torch.cdist(x, batch, p=2).pow(2).to(acc_dtype, non_blocking=True)
            logw = -dist2 / (two_sigma_squared + 1e-12)  # (Bx, Bd)

            # Update shift and rescale running sums
            batch_shift = logw.max(dim=1, keepdim=True).values  # (Bx,1)
            new_shift = torch.maximum(shift, batch_shift)
            # Rescale existing sums to new shift
            rescale = torch.exp(shift - new_shift)  # (Bx,1)
            numerator = numerator * rescale
            denominator = denominator * rescale

            # Compute stabilized weights for current batch
            weights = torch.exp(logw - new_shift)  # (Bx, Bd)

            # Weighted sum via matmul
            numerator = numerator + weights @ batch
            denominator = denominator + weights.sum(dim=1, keepdim=True)
            shift = new_shift

        x_denoised = numerator / denominator
        return x_denoised.view(shape)

    @staticmethod
    def _select_accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
        r"""
        Select appropriate accumulator dtype to avoid numerical issues.
        The distance computations can be computed faster in half precision
        """
        if dtype in [torch.float16, torch.bfloat16]:
            return torch.float32
        return dtype
