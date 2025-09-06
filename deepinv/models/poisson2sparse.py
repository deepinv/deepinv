# Adapted from https://github.com/tacalvin/Poisson2Sparse
# and https://github.com/drorsimon/CSCNet
from .base import Denoiser
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from deepinv.loss import Neighbor2Neighbor as N2N
from collections import namedtuple
from functools import wraps
from typing import Callable, Union
from deepinv.utils.compat import zip_strict


_ListaParams = namedtuple(
    "ListaParams", ["kernel_size", "num_filters", "stride", "unfoldings", "channels"]
)


class _SoftThreshold2d(nn.Module):
    """Learnable 2d channel-wise soft-thresholding layer

    :param int num_features: Number of channels in the input tensor
    :param float init_threshold: Initial value for the learned thresholds (default: 1e-2)
    """

    def __init__(self, num_features: int, *, init_threshold: float = 1e-2):
        super().__init__()
        self.thresholds = nn.Parameter(
            torch.full(size=(num_features,), fill_value=init_threshold),
        )

    def forward(self, x):
        # Channel-wise soft-thresholding with learned thresholds
        threshold = self.thresholds.view(1, -1, 1, 1)
        return self._soft_threshold(x, threshold=threshold)

    @staticmethod
    def _soft_threshold(x: torch.Tensor, *, threshold: Union[float, torch.Tensor]) -> torch.Tensor:
        """Soft-thresholding operation

        1. Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM Journal on Imaging Sciences, 2(1), 183–202. https://doi.org/10.1137/080716542

        :param torch.Tensor x: Input tensor
        :param Union[float, torch.Tensor] threshold: Threshold value (constant or per entry). If a tensor, it must be broadcastable to the shape of ``x``.
        :return: (:class:`torch.Tensor`) Soft-thresholded tensor
        """
        return (x.abs() - threshold).clamp(min=0.0) * x.sign()


# Credit goes to https://github.com/drorsimon/CSCNet
# Simon, Dror, and Michael Elad. "Rethinking the CSC model for natural images." Advances in Neural Information Processing Systems 32 (2019).


class ConvLista(nn.Module):
    def __init__(
        self,
        *,
        kernel_size,
        num_filters,
        stride,
        num_iter,
        channels,
        A=None,
        B=None,
        C=None,
        threshold=1e-2,
        norm=False,
    ):
        super().__init__()
        params = _ListaParams(
            kernel_size,
            num_filters,
            stride,
            num_iter,
            channels,
        )
        if A is None:
            A = torch.randn(
                params.num_filters,
                params.channels,
                params.kernel_size,
                params.kernel_size,
            )
            l = conv_power_method(A, [128, 128], num_iters=20, stride=params.stride)
            A /= torch.sqrt(l)
        if B is None:
            B = torch.clone(A)
        if C is None:
            C = torch.clone(A)
        self.apply_A = torch.nn.ConvTranspose2d(
            params.num_filters,
            params.channels,
            kernel_size=params.kernel_size,
            stride=params.stride,
            bias=False,
        )
        self.apply_B = torch.nn.Conv2d(
            params.channels,
            params.num_filters,
            kernel_size=params.kernel_size,
            stride=params.stride,
            bias=False,
        )
        self.apply_C = torch.nn.ConvTranspose2d(
            params.num_filters,
            params.channels,
            kernel_size=params.kernel_size,
            stride=params.stride,
            bias=False,
        )
        self.apply_A.weight.data = A
        self.apply_B.weight.data = B
        self.apply_C.weight.data = C
        self.soft_threshold = _SoftThreshold2d(
            params.num_filters,
            init_threshold=threshold
        )
        self.params = params
        self.num_iter = params.unfoldings


    @staticmethod
    def _calc_pad_sizes(x: torch.Tensor, *, kernel_size: int, stride: int) -> tuple[int, int, int, int]:
        return (
            stride,
            (
                stride
                if (x.shape[3] + stride - kernel_size) % stride == 0
                else 2 * stride - ((x.shape[3] + stride - kernel_size) % stride)
            ),
            stride,
            (
                stride
                if (x.shape[2] + stride - kernel_size) % stride == 0
                else 2 * stride - ((x.shape[2] + stride - kernel_size) % stride)
            )
        )


    @classmethod
    def _augment_images(cls, y: torch.Tensor, *, stride: int, kernel_size: int):
        if stride == 1:
            mask = torch.ones_like(y)
        elif stride > 1:
            left_pad, right_pad, top_pad, bot_pad = cls._calc_pad_sizes(
                y, kernel_size=kernel_size, stride=stride
            )
            augmented_y = torch.empty((
                y.shape[0],
                stride**2,
                y.shape[1],
                y.shape[2] + top_pad + bot_pad,
                y.shape[3] + left_pad + right_pad,
            ), dtype=y.dtype, device=y.device)
            mask = torch.zeros_like(augmented_y)
            for num, (augmented_y_part, mask_part) in enumerate(
                    zip_strict(augmented_y.unbind(dim=1), mask.unbind(dim=1))
                ):
                row_shift = num // stride
                col_shift = num % stride
                augmented_y_part.copy_(
                    F.pad(
                        y,
                        pad=(
                            left_pad - col_shift,
                            right_pad + col_shift,
                            top_pad - row_shift,
                            bot_pad + row_shift,
                        ),
                        mode="reflect",
                    )
                )
                mask_part[
                    ...,
                    top_pad - row_shift : y.shape[2] + top_pad - row_shift,
                    left_pad - col_shift : y.shape[3] + left_pad - col_shift,
                ].fill_(1.0)
            y = augmented_y.flatten(start_dim=0, end_dim=1)
            mask = mask.flatten(start_dim=0, end_dim=1)
        else:
            raise ValueError("Stride must be a positive integer.")
        return y, mask


    def forward(self, y: torch.Tensor) -> torch.Tensor:
        stride = self.params.stride
        shape = y.shape
        # We apply the algorithm to every possible shift of the input image
        # depending on the stride in order to have a result that does not
        # depend on the alignment of the input image. If the stride is 1, y is
        # not modified.
        y, mask = self._augment_images(y, stride=stride, kernel_size=self.params.kernel_size)

        # NOTE: \Gamma is initialized as \Gamma_0 = 0 and for efficiency we start at
        # \Gamma_1 = S_\tau(BY) by applying \Gamma_{k+1} = S_\tau(\Gamma_k + B(Y - A\Gamma_k))
        # and only compute T - 1 iterations instead of starting at \Gamma_0 and doing T iterations.
        # For more details, see Section 4.2 in https://doi.org/10.48550/arXiv.1909.05742
        gamma = self.soft_threshold(self.apply_B(y))
        for _ in range(self.num_iter - 1):
            # Eq. (15) in https://doi.org/10.48550/arXiv.1909.05742
            # \Gamma_{k+1} = S_\tau(\Gamma_k + B(Y - A\Gamma_k))
            gamma = self.soft_threshold(gamma + self.apply_B(y - self.apply_A(gamma)))
        # \hat X = C\Gamma_T
        x = self.apply_C(gamma)

        # Deal with the shifted copies
        x = torch.masked_select(x, mask.bool())
        x = x.reshape(
            shape[0], stride**2, *shape[1:]
        )
        # Average over all the shifts
        x = x.mean(dim=1, keepdim=False)

        # Post-processing
        return torch.clamp(x, min=0.0, max=1.0)


def conv_power_method(D, image_size, num_iters=100, stride=1):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    needles_shape = [
        int(((image_size[0] - D.shape[-2]) / stride) + 1),
        int(((image_size[1] - D.shape[-1]) / stride) + 1),
    ]
    x = torch.randn(1, D.shape[0], *needles_shape).type_as(D)
    for _ in range(num_iters):
        c = torch.norm(x.reshape(-1))
        x = x / c
        y = F.conv_transpose2d(x, D, stride=stride)
        x = F.conv2d(y, D, stride=stride)
    return torch.norm(x.reshape(-1))


class _Poisson2SparseLoss(nn.Module):
    def __init__(self, *, weight_n2n, weight_l1_regularization):
        super().__init__()
        self.pll_loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
        self.weight_n2n = weight_n2n
        self.weight_l1_regularization = weight_l1_regularization

    def forward(self, *, y, model):
        # Stop gradients
        with torch.no_grad():
            x_hat = model(y).detach()

        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        mask1, mask2 = N2N.generate_mask_pair(y)

        y1 = N2N.generate_subimages(y, mask1)
        y2 = N2N.generate_subimages(y, mask2)

        x_hat1 = N2N.generate_subimages(x_hat, mask1)
        x_hat2 = N2N.generate_subimages(x_hat, mask2)

        y2_hat = model(y1)
        y2_hat = torch.clamp(y2_hat, 0.0, 1.0)

        # $\mathcal L_{\text{Poisson}}$
        loss = self.pll_loss_fn(y2_hat, y2)

        # $\mathcal L_{\mathrm{L1}}$
        loss += F.l1_loss(y2_hat, y2)

        # $\mathcal L_{\mathrm{N}}$
        loss += self.weight_n2n * (((y2_hat - y2) - (x_hat1 - x_hat2)) ** 2).mean()

        # Sparsity-inducing $\ell^1$ weight-regularization
        l1_regularization = 0.0
        for param in model.parameters():
            l1_regularization += param.abs().sum()
        l1_regularization = self.weight_l1_regularization * l1_regularization
        loss += l1_regularization

        return loss, x_hat


def _pad_fn_even(func: Callable, *, value: float = 0.0):
    def _decorate(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            y = args[1]
            H, W = y.shape[-2:]
            x_pad = F.pad(y, (0, H % 2, 0, H % 2), value=value)
            x_hat = fn(args[0], x_pad, *args[2:], **kwargs)
            return x_hat[..., :H, :W]

        return wrapper

    return _decorate(func)


class Poisson2Sparse(Denoiser):
    def __init__(
            self, *, backbone: torch.nn.Module, lr: float, weight_n2n: float, weight_l1_regularization: float, num_iter: int, verbose: bool
    ):
        super().__init__()
        self.backbone = backbone
        self.lr = lr
        self.weight_n2n = weight_n2n
        self.weight_l1_regularization = weight_l1_regularization
        self.num_iter = num_iter
        self.verbose = verbose
        self._loss_fn = _Poisson2SparseLoss(
            weight_n2n=self.weight_n2n,
            weight_l1_regularization=self.weight_l1_regularization,
        )

    @_pad_fn_even
    def forward(self, y, physics=None):
        backbone = self.backbone
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=self.lr)

        x_hat_avg = None
        for _ in trange(self.num_iter, disable=not self.verbose):
            loss, x_hat = self._loss_fn(y=y, model=backbone)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                x_hat = x_hat
                if x_hat_avg is None:
                    x_hat_avg = x_hat
                else:
                    exp_weight = 0.98
                    x_hat_avg = x_hat_avg * exp_weight + x_hat * (1 - exp_weight)
                x_hat_avg = x_hat_avg.detach()

        return x_hat_avg
