from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm import trange
from functools import wraps
from typing import Callable
from deepinv.models.base import Denoiser
from deepinv.utils.compat import zip_strict


class ConvLista(nn.Module):
    r"""
    Convolutional LISTA network.

    The architecture was introduced by :footcite:t:`simon2019rethinking`, and it is well suited as a backbone for Poisson2Sparse (see :class:`deepinv.models.Poisson2Sparse`).

    .. note::

        The decoder expects images with a dynamic range normalized between zero and one.

    :param int in_channels: Number of channels in the input image.
    :param int out_channels: Number of channels in the output image.
    :param int kernel_size: Size of the convolutional kernels (default: 3).
    :param int num_filters: Number of filters in the convolutional layers (default: 512).
    :param int stride: Stride of the convolutional layers (default: 2).
    :param int num_iter: Number of iterations of the LISTA algorithm (default: 10).
    :param float threshold: Initial value for the learned soft-thresholding (default: 1e-2).

    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_filters: int = 512,
        stride: int = 2,
        num_iter: int = 10,
        threshold: float = 1e-2,
    ):
        super().__init__()
        conv_A = torch.nn.Conv2d(
            in_channels,
            num_filters,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        conv_B = torch.nn.ConvTranspose2d(
            num_filters,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        conv_C = torch.nn.ConvTranspose2d(
            num_filters,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

        # Initialize the convolutional layers as the spectral normalization of
        # convolutions with kernel entries sampled iid from the standard
        # Gaussian distribution.
        with torch.no_grad():
            for conv in [conv_A, conv_B, conv_C]:
                init.normal_(conv.weight)
                sqnorm = self._conv2d_spectral_sqnorm(
                    conv.weight, image_size=[128, 128], num_iters=20, stride=stride
                )
                conv.weight /= torch.sqrt(sqnorm)

        self.conv_A = conv_A
        self.conv_B = conv_B
        self.conv_C = conv_C
        self.soft_threshold = self._SoftThreshold2d(
            num_filters, init_threshold=threshold
        )

        self.stride = stride
        self.kernel_size = kernel_size
        self.num_iter = num_iter

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        stride = self.stride
        shape = y.shape
        # We apply the algorithm to every possible shift of the input image
        # depending on the stride in order to have a result that does not
        # depend on the alignment of the input image. If the stride is 1, y is
        # not modified.
        y, mask = self._shifts_augmentation(
            y, stride=stride, kernel_size=self.kernel_size
        )

        # NOTE: \Gamma is initialized as \Gamma_0 = 0 and for efficiency we start at
        # \Gamma_1 = S_\tau(BY) by applying \Gamma_{k+1} = S_\tau(\Gamma_k + A(Y - B\Gamma_k))
        # and only compute T - 1 iterations instead of starting at \Gamma_0 and doing T iterations.
        # For more details, see Section 4.2 in https://doi.org/10.48550/arXiv.1909.05742
        gamma = self.soft_threshold(self.conv_A(y))
        for _ in range(self.num_iter - 1):
            # Eq. (15) in https://doi.org/10.48550/arXiv.1909.05742
            # \Gamma_{k+1} = S_\tau(\Gamma_k + A(Y - B\Gamma_k))
            gamma = self.soft_threshold(gamma + self.conv_A(y - self.conv_B(gamma)))
        # \hat X = C\Gamma_T
        x = self.conv_C(gamma)

        # Deal with the shifted copies
        x = torch.masked_select(x, mask.bool())
        x = x.reshape(shape[0], stride**2, *shape[1:])
        # Average over all the shifts
        x = x.mean(dim=1, keepdim=False)

        # Post-processing
        return torch.clamp(x, min=0.0, max=1.0)

    @classmethod
    def _shifts_augmentation(cls, y: torch.Tensor, *, stride: int, kernel_size: int):
        if stride == 1:
            mask = torch.ones_like(y)
        elif stride > 1:
            left_pad, right_pad, top_pad, bot_pad = (
                stride,
                (
                    stride
                    if (y.shape[3] + stride - kernel_size) % stride == 0
                    else 2 * stride - ((y.shape[3] + stride - kernel_size) % stride)
                ),
                stride,
                (
                    stride
                    if (y.shape[2] + stride - kernel_size) % stride == 0
                    else 2 * stride - ((y.shape[2] + stride - kernel_size) % stride)
                ),
            )
            augmented_y = torch.empty(
                (
                    y.shape[0],
                    stride**2,
                    y.shape[1],
                    y.shape[-2] + top_pad + bot_pad,
                    y.shape[-1] + left_pad + right_pad,
                ),
                dtype=y.dtype,
                device=y.device,
            )
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

    @staticmethod
    def _conv2d_spectral_sqnorm(
        weight: torch.Tensor,
        *,
        image_size: tuple[int, int],
        num_iters: int = 100,
        stride: int = 1,
    ):
        """
        Compute the squared spectral norm of a 2d convolutional layer

        :param torch.Tensor weight: Weights of the convolutional layer
        :param tuple[int, int] image_size: The image size used for inputs
        :param int num_iters: The number of iterations to run the power method
        :param int stride: The stride of the convolutional layer
        :return: The squared spectral norm
        """
        needles_shape = [
            int(((image_size[0] - weight.shape[-2]) / stride) + 1),
            int(((image_size[1] - weight.shape[-1]) / stride) + 1),
        ]
        x = torch.randn(1, weight.shape[0], *needles_shape).type_as(weight)
        for _ in range(num_iters):
            c = torch.linalg.vector_norm(x.reshape(-1), ord=2)
            x = x / c
            y = F.conv_transpose2d(x, weight, stride=stride)
            x = F.conv2d(y, weight, stride=stride)
        return torch.linalg.vector_norm(x.reshape(-1), ord=2)

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
        def _soft_threshold(
            x: torch.Tensor, *, threshold: float | torch.Tensor
        ) -> torch.Tensor:
            """Soft-thresholding operation

            :param torch.Tensor x: Input tensor
            :param float | torch.Tensor threshold: Threshold value (constant or per entry). If a tensor, it must be broadcastable to the shape of ``x``.
            :return: (:class:`torch.Tensor`) Soft-thresholded tensor
            """
            return (x.abs() - threshold).clamp(min=0.0) * x.sign()


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
    r"""
    Poisson2Sparse model for Poisson denoising.

    This method, introduced by :footcite:t:`ta2022poisson2sparse`, reconstructs an image corrupted by Poisson noise by learning a sparse non-linear dictionary parametrized by a neural network using a combination of Neighbor2Neighbor :footcite:t:`huang2021neighbor2neighbor`, of the negative log Poisson likelihood, of the :math:`\ell^1` pixel distance and of a sparsity-inducing :math:`\ell^1` regularization function on the weights.

    .. note::

        This method does not use knowledge of the physics model and assumes a Poisson degradation model internally. Therefore, the physics object can be omitted when calling the model and specifying it will have no effect.

    .. note::

        The denoiser expects images with a dynamic range normalized between zero and one.

    :param torch.nn.Module, None backbone: Neural network used as a non-linear dictionary. If ``None``, a default :class:`deepinv.models.ConvLista` model is used.
    :param float lr: Learning rate of the AdamW optimizer (default: 1e-4).
    :param float weight_n2n: Weight of the Neighbor2Neighbor loss term (default: 2.0).
    :param float weight_l1_regularization: Weight of the sparsity-inducing :math:`\ell^1` regularization on the weights (default: 1e-5).
    :param int num_iter: Number of optimization iterations (default: 200).
    :param bool verbose: If ``True``, print progress (default: ``False``).

    """

    def __init__(
        self,
        backbone: torch.nn.Module | None = None,
        *,
        lr: float = 1e-4,
        weight_n2n: float = 2.0,
        weight_l1_regularization: float = 1e-5,
        num_iter: int = 200,
        verbose: bool = False,
    ):
        super().__init__()
        if backbone is None:
            backbone = ConvLista(in_channels=1, out_channels=1)
        self.backbone = backbone
        self.lr = lr
        self.weight_n2n = weight_n2n
        self.weight_l1_regularization = weight_l1_regularization
        self.num_iter = num_iter
        self.verbose = verbose
        self._loss_fn = self._Poisson2SparseLoss(
            weight_n2n=self.weight_n2n,
            weight_l1_regularization=self.weight_l1_regularization,
        )

    @_pad_fn_even
    def forward(self, y, physics=None, **kwargs):
        backbone = self.backbone
        optimizer = torch.optim.AdamW(backbone.parameters(), lr=self.lr)

        x_hat_avg = None
        for _ in trange(self.num_iter, disable=not self.verbose):
            loss, x_hat = self._loss_fn(y=y, model=backbone)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                if x_hat_avg is None:
                    x_hat_avg = x_hat.detach()
                else:
                    exp_weight = 0.98
                    x_hat_avg = x_hat_avg * exp_weight + x_hat * (1 - exp_weight)

        return x_hat_avg

    class _Poisson2SparseLoss(nn.Module):
        def __init__(self, *, weight_n2n, weight_l1_regularization):
            super().__init__()
            self.pll_loss_fn = nn.PoissonNLLLoss(log_input=False, full=True)
            self.weight_n2n = weight_n2n
            self.weight_l1_regularization = weight_l1_regularization

        def forward(self, *, y, model):
            from deepinv.loss import Neighbor2Neighbor as N2N

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
