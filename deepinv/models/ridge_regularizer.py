from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_weights_url

class RidgeRegularizer(nn.Module):
    r"""
    (Weakly) Convex Ridge Regularizer model :math:`\reg{x}=\sum_{c} \psi_c(W_c x)`.

    for filters :math:`W_c` and potentials :math:`\psi_c`. The filters :math:`W_c` are realized by a concatenation multiple convolution
    layers without nonlinearity. The potentials :math:`\psi_c` are given by scaled versions smoothed absolute values,
    see :footcite:t:`hertrich2025learning` for a precise description.

    To allow the automatic tuning of the regularization parameter, we parameterize the regularizer with two additional scalings, i.e.,
    we implement :math:`\frac{\alpha}{\sigma^2}\reg{\sigma x}` instead of :math:`\reg{x}` where :math:`\alpha` and :math:`\sigma` are learnable parameters of the regularizer.
    If the weak CRR is used, :math:`\alpha` is fixed per default, since it changes the weak convexity constant.

    The (W)CRR was introduced by :footcite:t:`goujon2023neural` and :footcite:t:`goujon2024learning`.
    The specific implementation is taken from :footcite:t:`hertrich2025learning`.

    This model can be used as a prior through :class:`deepinv.optim.RidgeRegularizer`.

    :param int in_channels: Number of input channels (`1` for gray valued images, `3` for color images). Default: `3`
    :param float weak_convexity: Weak convexity of the regularizer. Set to `0.0` for a convex regularizer and to `1.0` for a 1-weakly convex regularizer.
        Default: `0.0`
    :param list of int nb_channels: List of ints taking the hidden number of channels in the multiconvolution. Default: `[4, 8, 64]`
    :param list of int filter_sizes: List of ints taking the kernel sizes of the convolution. Default: `[5,5,5]`
    :param str device: Device for the weights. Default: `"cpu"`
    :param str, None pretrained: use pretrained weights. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with 3 or 1 input/output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-learned-reg>` for more details.
    :param bool warn_output_scaling: warn if `weak_convexity>0` and the output scaling (:math:`\log(\alpha)` in the above description) is not zero. This case
        destroys the weak convexity constant defined by teh `weak_convexity` argument. Default: `True`
    """

    def __init__(
        self,
        in_channels: int = 3,
        weak_convexity: float = 0.0,
        nb_channels: tuple[int, ...] = (4, 8, 64),
        filter_sizes: tuple[int, ...] = (5, 5, 5),
        device: str = "cpu",
        pretrained: str | None = "download",
        warn_output_scaling: bool = True,
    ):
        super().__init__()
        nb_channels = [in_channels] + list(nb_channels)
        self.warn_output_scaling = warn_output_scaling
        self.nb_filters = nb_channels[-1]
        self.filter_size = sum(filter_sizes) - len(filter_sizes) + 1
        self.filters = nn.Sequential(
            *[
                nn.Conv2d(
                    nb_channels[i],
                    nb_channels[i + 1],
                    filter_sizes[i],
                    padding=filter_sizes[i] // 2,
                    bias=False,
                    device=device,
                )
                for i in range(len(filter_sizes))
            ]
        )

        class ZeroMean(nn.Module):
            """Enforces zero mean on the filters"""

            def forward(self, x):
                return x - torch.mean(x, dim=(1, 2, 3), keepdim=True)

        torch.nn.utils.parametrize.register_parametrization(
            self.filters[0], "weight", ZeroMean()
        )

        self.dirac = torch.zeros(
            (1, in_channels, 2 * self.filter_size - 1, 2 * self.filter_size - 1),
            device=device,
        )
        self.dirac[0, 0, self.filter_size - 1, self.filter_size - 1] = 1.0

        self.scaling = nn.Parameter(
            torch.log(torch.tensor(20.0, device=device))
            * torch.ones((1, self.nb_filters, 1, 1), device=device)
        )
        self.input_scaling = nn.Parameter(torch.tensor(0.0, device=device))
        self.beta = nn.Parameter(torch.tensor(4.0, device=device))
        self.output_scaling = nn.Parameter(torch.tensor(0.0, device=device)).requires_grad_(
            weak_convexity == 0.0
        )
        self.weak_cvx = weak_convexity

        if pretrained is not None:
            if pretrained == "download":
                if in_channels == 1 and weak_convexity == 0.0:
                    file_name = "CRR_gray.pt"
                elif in_channels == 3 and weak_convexity == 0.0:
                    file_name = "CRR_color.pt"
                elif in_channels == 1 and weak_convexity == 1.0:
                    file_name = "WCRR_gray.pt"
                elif in_channels == 3 and weak_convexity == 1.0:
                    file_name = "WCRR_color.pt"
                else:
                    raise ValueError(
                        "Weights are only available for weak_convexity equal to 0.0 or 1.0 and in_channels in [1, 3]!"
                    )
                url = get_weights_url(model_name="RidgeRegularizer ", file_name=file_name)
                ckpt = torch.hub.load_state_dict_from_url(
                    url,
                    map_location=lambda storage, loc: storage,
                    file_name=file_name,
                )
                self.load_state_dict(ckpt, strict=True)
            else:
                self.load_state_dict(torch.load(pretrained, map_location=device))

    def __smooth_l1(self, x):
        return torch.clip(x**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(x), 1.0) - 1.0

    def __grad_smooth_l1(self, x):
        return torch.clip(x, -1.0, 1.0)

    def __get_conv_lip(self):
        impulse = self.filters(self.dirac)
        for filt in reversed(self.filters):
            impulse = F.conv_transpose2d(impulse, filt.weight, padding=filt.padding)
        return torch.fft.rfft2(impulse, s=[256, 256]).abs().max()

    def __conv(self, x):
        x = x / torch.sqrt(self.__get_conv_lip())
        return self.filters(x)

    def __conv_transpose(self, x):
        x = x / torch.sqrt(self.__get_conv_lip())
        for filt in reversed(self.filters):
            x = F.conv_transpose2d(x, filt.weight, padding=filt.padding)
        return x

    def grad(self, x, *args, get_energy=False, **kwargs):
        r"""
        Calculates the gradient of the regularizer at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param bool get_energy: Optional flag. If set to True, the function additionally returns the objective value at :math:`x`. Dafault: False.
        :return: (:class:`torch.Tensor`) gradient at :math:`x`.
        """
        grad = self.__conv(x)
        grad = grad * torch.exp(self.scaling + self.input_scaling)
        if get_energy:
            reg = (
                self.__smooth_l1(torch.exp(self.beta) * grad) * torch.exp(-self.beta)
                - self.__smooth_l1(grad) * self.weak_cvx
            )
            reg = reg * torch.exp(
                self.output_scaling - 2 * self.scaling - 2 * self.input_scaling
            )
            reg = reg.sum(dim=(1, 2, 3))
        grad = (
            self.__grad_smooth_l1(torch.exp(self.beta) * grad)
            - self.__grad_smooth_l1(grad) * self.weak_cvx
        )
        grad = grad * torch.exp(self.output_scaling - self.scaling - self.input_scaling)
        grad = self.__conv_transpose(grad)
        if get_energy:
            return reg, grad
        return grad

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the regularizer :math:`\reg{x}=\sum_{c} \psi_c(W_c x)`

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        if (
            not self.output_scaling == 0.0
            and not self.weak_cvx == 0
            and self.warn_output_scaling
        ):
            warnings.warn(
                "The parameter RidgeRegularizer.output_scaling is not zero even though RidgeRegularizer.weak_convexity is not zero! "
                + "This means that the weak convexity parameter of the RidgeRegularizer is not RidgeRegularizer.weak_convexity but exp(output_scaling)*RidgeRegularizer.weak_convexity. "
                + "If you require the RidgeRegularizer to keep the weak convexity, set RidgeRegularizer.output_scaling.requires_grad_(False) for all training methods and do not "
                + "change RidgeRegularizer.output_scaling. To suppress this warning, set warn_output_scaling in the constructor of the RidgeRegularizer to False."
            )
        reg = self.__conv(x)
        reg = reg * torch.exp(self.scaling + self.input_scaling)
        reg = (
            self.__smooth_l1(torch.exp(self.beta) * reg) * torch.exp(-self.beta)
            - self.__smooth_l1(reg) * self.weak_cvx
        )
        reg = reg * torch.exp(
            self.output_scaling - 2 * self.scaling - 2 * self.input_scaling
        )
        reg = reg.sum(dim=(1, 2, 3))
        return reg

    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        return super()._apply(fn)