from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepinv.optim.potential import Potential
from deepinv.models.tv import TVDenoiser
from deepinv.models.wavdict import WaveletDenoiser, WaveletDictDenoiser
from deepinv.utils import patch_extractor
from deepinv.models.utils import get_weights_url, load_state_dict_from_url

if TYPE_CHECKING:
    from deepinv.optim import Prior


class Prior(Potential):
    r"""
    Prior term :math:`\reg{x}`.

    This is the base class for the prior term :math:`\reg{x}`. As a child class from the Poential class, it comes with methods for computing
    :math:`\operatorname{prox}_{g}` and :math:`\nabla \regname`.
    To implement a custom prior, for an explicit prior, overwrite :math:`\regname` (do not forget to specify
    `self.explicit_prior = True`)

    This base class is also used to implement implicit priors. For instance, in PnP methods, the method computing the
    proximity operator is overwritten by a method performing denoising. For an implicit prior, overwrite `grad`
    or `prox`.


    .. note::

        The methods for computing the proximity operator and the gradient of the prior rely on automatic
        differentiation. These methods should not be used when the prior is not differentiable, although they will
        not raise an error.


    :param Callable g: Prior function :math:`g(x)`.
    """

    def __init__(self, g: Prior = None, *args, **kwargs):
        super().__init__(*args, fn=g, **kwargs)
        self.explicit_prior = False if self._fn is None else True


class ZeroPrior(Prior):
    r"""
    Zero prior :math:`\reg{x} = 0`.
    """

    def __init__(self):
        super().__init__()
        self.explicit_prior = True

    def fn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the zero prior :math:`\reg{x} = 0` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        return torch.zeros(x.shape[0], device=x.device)

    def grad(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the gradient of the zero prior :math:`\reg{x} = 0` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) gradient at :math:`x`.
        """
        return torch.zeros_like(x)

    def prox(
        self, x: torch.Tensor, ths: float = 1.0, gamma: float = 1.0, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the proximal operator of the zero prior :math:`\reg{x} = 0` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        return x


class PnP(Prior):
    r"""
    Plug-and-play prior :math:`\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x)`.


    :param Callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    """

    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.explicit_prior = False

    def prox(
        self, x: torch.Tensor, sigma_denoiser: float, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Uses denoising as the proximity operator of the PnP prior :math:`\regname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float sigma_denoiser: noise level parameter of the denoiser.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        return self.denoiser(x, sigma_denoiser)


class RED(Prior):
    r"""
    Regularization-by-Denoising (RED) prior :math:`\nabla \reg{x} = x - \operatorname{D}_{\sigma}(x)`.


    :param Callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    """

    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.explicit_prior = False

    def grad(
        self, x: torch.Tensor, sigma_denoiser: float, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Calculates the gradient of the prior term :math:`\regname` at :math:`x`.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x g`, computed in :math:`x`.
        """
        return x - self.denoiser(x, sigma_denoiser)


class ScorePrior(Prior):
    r"""
    Score via MMSE denoiser :math:`\nabla \reg{x}=\left(x-\operatorname{D}_{\sigma}(x)\right)/\sigma^2`.

    This approximates the score of a distribution using Tweedie's formula, i.e.,

    .. math::

        - \nabla \log p_{\sigma}(x) \propto \left(x-D(x,\sigma)\right)/\sigma^2

    where :math:`p_{\sigma} = p*\mathcal{N}(0,I\sigma^2)` is the prior convolved with a Gaussian kernel,
    :math:`D(\cdot,\sigma)` is a (trained or model-based) denoiser with noise level :math:`\sigma`,
    which is typically set to a low value.

    .. note::

        If :math:`\sigma=1`, this prior is equal to :class:`deepinv.optim.RED`, which is defined in
        Regularization by Denoising (RED) :footcite:t:`romano2017little` and doesn't require the normalization.


    .. note::

        This class can also be used with maximum-a-posteriori (MAP) denoisers,
        but :math:`p_{\sigma}(x)` is not given by the convolution with a Gaussian kernel, but rather
        given by the Moreau-Yosida envelope of :math:`p(x)`, i.e.,

        .. math::

            p_{\sigma}(x)=e^{- \inf_z \left(-\log p(z) + \frac{1}{2\sigma}\|x-z\|^2 \right)}.
    """

    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.explicit_prior = False

    def grad(
        self, x: torch.Tensor, sigma_denoiser: float, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Applies the denoiser to the input signal.

        :param torch.Tensor x: the input tensor.
        :param float sigma_denoiser: the noise level.
        :return: (torch.Tensor) gradient at :math:`x`.
        """
        return self.stable_division(
            x - self.denoiser(x, sigma_denoiser, *args, **kwargs), sigma_denoiser**2
        )

    def score(
        self, x: torch.Tensor, sigma_denoiser: float, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the score function :math:`\nabla \log p_\sigma`, using Tweedie's formula.

        :param torch.Tensor x: the input tensor.
        :param float sigma_denoiser: the noise level.
        """
        return self.stable_division(
            self.denoiser(x, sigma_denoiser, *args, **kwargs) - x, sigma_denoiser**2
        )

    @staticmethod
    def stable_division(
        a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-7
    ) -> torch.Tensor:
        r"""
        Performs a safe-guarded division by adding a small constant :math:`\epsilon` to the denominator when it is close to zero.

        :param torch.Tensor a: numerator.
        :param torch.Tensor b: denominator.
        :param float epsilon: small constant added to the denominator when it is close to zero.

        :return: (:class:`torch.Tensor`) result of the division.
        """

        if isinstance(b, torch.Tensor):
            b = torch.where(
                b.abs().detach() > epsilon,
                b,
                torch.full_like(b, fill_value=epsilon) * b.sign(),
            )
        elif isinstance(b, (float, int)):
            b = max(epsilon, abs(b)) * np.sign(b)

        return a / b


class Tikhonov(Prior):
    r"""
    Tikhonov regularizer :math:`\reg{x} = \frac{1}{2}\| x \|_2^2`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def fn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the Tikhonov regularizer :math:`\reg{x} = \frac{1}{2}\| x \|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        return (
            0.5 * torch.linalg.vector_norm(x, dim=tuple(range(1, x.dim())), ord=2) ** 2
        )

    def grad(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Calculates the gradient of the Tikhonov regularization term :math:`\regname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (:class:`torch.Tensor`) gradient at :math:`x`.
        """
        return x

    def prox(
        self, x: torch.Tensor, *args, gamma: float = 1.0, **kwargs
    ) -> torch.Tensor:
        r"""
        Calculates the proximity operator of the Tikhonov regularization term :math:`\gamma g` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        return (1 / (gamma + 1)) * x


class L1Prior(Prior):
    r"""
    :math:`\ell_1` prior :math:`\reg{x} = \| x \|_1`.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def fn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the regularizer :math:`\reg{x} = \| x \|_1`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        return torch.linalg.vector_norm(x, ord=1, dim=tuple(range(1, x.dim())))

    def prox(
        self, x: torch.Tensor, *args, ths: float = 1.0, gamma: float = 1.0, **kwargs
    ) -> torch.Tensor:
        r"""
        Calculates the proximity operator of the l1 regularization term :math:`\regname` at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = \operatorname{sign}(x) \max(|x| - \gamma, 0)


        where :math:`\gamma` is a stepsize.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        lambd = ths * gamma
        if isinstance(lambd, float):
            return torch.nn.functional.softshrink(
                x, lambd=lambd
            )  # this is faster but not batchable on lambd.
        else:
            return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - lambd)


class WaveletPrior(Prior):
    r"""
    Wavelet prior :math:`\reg{x} = \|\Psi x\|_{p}`.

    :math:`\Psi` is an orthonormal wavelet transform, and :math:`\|\cdot\|_{p}` is the :math:`p`-norm, with
    :math:`p=0`, :math:`p=1`, or :math:`p=\infty`.

    If clamping parameters are provided, the prior writes as :math:`\reg{x} = \|\Psi x\|_{p} + \iota_{[c_{\text{min}}, c_{\text{max}}]}(x)`,
    where :math:`\iota_{[c_{\text{min}}, c_{\text{max}}]}(x)` is the indicator function of the interval :math:`[c_{\text{min}}, c_{\text{max}}]`.

    .. note::
        Following common practice in signal processing, only detail coefficients are regularized, and the approximation
        coefficients are left untouched.

    .. warning::
        For 3D data, the computational complexity of the wavelet transform cubically with the size of the support. For
        large 3D data, it is recommended to use wavelets with small support (e.g. db1 to db4).


    :param int level: level of the wavelet transform. Default is 3.
    :param str wv: wavelet name to choose among those available in `pywt <https://pywavelets.readthedocs.io/en/latest/>`_. Default is "db8".
    :param float p: :math:`p`-norm of the prior. Default is 1.
    :param str device: device on which the wavelet transform is computed. Default is "cpu".
    :param int wvdim: dimension of the wavelet transform, can be either 2 or 3. Default is 2.
    :param bool is_complex: whether the input is complex-valued. Default is False.
    :param str mode: padding mode for the wavelet transform (default: "zero").
    :param float clamp_min: minimum value for the clamping. Default is None.
    :param float clamp_max: maximum value for the clamping. Default is None.
    """

    def __init__(
        self,
        level: int = 3,
        wv: str = "db8",
        p: float = 1,
        device: str = "cpu",
        wvdim: int = 2,
        is_complex: bool = False,
        mode: str = "zero",
        clamp_min: float = None,
        clamp_max: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.p = p
        self.wv = wv
        self.wvdim = wvdim
        self.level = level
        self.device = device
        self.mode = mode
        self.is_complex = is_complex

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        if p == 0:
            self.non_linearity = "hard"
        elif p == 1:
            self.non_linearity = "soft"
        elif p == np.inf or p == "inf":
            self.non_linearity = "topk"
        else:
            raise ValueError("p should be 0, 1 or inf")

        if type(self.wv) == str:
            self.WaveletDenoiser = WaveletDenoiser(
                level=self.level,
                wv=self.wv,
                device=self.device,
                non_linearity=self.non_linearity,
                is_complex=self.is_complex,
                wvdim=self.wvdim,
            )
        elif type(self.wv) == list:
            self.WaveletDenoiser = WaveletDictDenoiser(
                level=self.level,
                list_wv=self.wv,
                max_iter=10,
                non_linearity=self.non_linearity,
                is_complex=self.is_complex,
                wvdim=self.wvdim,
            )
        else:
            raise ValueError(
                f"wv should be a string (name of the wavelet) or a list of strings (list of wavelet names). Got {type(self.wv)} instead."
            )

    def fn(self, x: torch.Tensor, *args, reduce: bool = True, **kwargs) -> torch.Tensor:
        r"""
        Computes the regularizer

        .. math::
            \begin{equation}
             {\regname}_{i,j}(x) = \|(\Psi x)_{i,j}\|_{p}
             \end{equation}


        where :math:`\Psi` is an orthonormal wavelet transform, :math:`i` and :math:`j` are the indices of the
        wavelet sub-bands,  and :math:`\|\cdot\|_{p}` is the :math:`p`-norm, with
        :math:`p=0`, :math:`p=1`, or :math:`p=\infty`. As mentioned in the class description, only detail coefficients
        are regularized, and the approximation coefficients are left untouched.

        If `reduce` is set to `True`, the regularizer is summed over all detail coefficients, yielding

        .. math::
                \regname(x) = \|\Psi x\|_{p}.

        If `reduce` is set to `False`, the regularizer is returned as a list of the norms of the detail coefficients.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :param bool reduce: if True, the prior is summed over all detail coefficients. Default is True.
        :return: (:class:`torch.Tensor`) prior :math:`g(x)`.
        """
        list_dec = self.psi(x)
        list_norm = torch.cat(
            [
                torch.linalg.norm(dec, ord=self.p, dim=1, keepdim=True)
                for dec in list_dec
            ],
            dim=1,
        )
        if reduce:
            return torch.sum(list_norm, dim=1)
        else:
            return list_norm

    def prox(
        self, x: torch.Tensor, *args, ths: float = 0.1, gamma: float = 1.0, **kwargs
    ) -> torch.Tensor:
        r"""Compute the proximity operator of the wavelet prior with the denoiser :class:`~deepinv.models.WaveletDenoiser`.
        Only detail coefficients are thresholded.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param int, float, torch.Tensor ths: thresholding parameter :math:`\gamma`.
            If `ths` is a tensor, it should be of shape
            ``(B,)`` (same coefficent for all levels), ``(B, n_levels-1)`` (one coefficient per level),
            or ``(B, n_levels-1, 3)`` (one coefficient per subband and per level). `B` should be the same as the batch size of the input or `1`.
            If ``non_linearity`` equals ``"soft"`` or ``"hard"``, ``ths`` serves as a (soft or hard)
            thresholding parameter for the wavelet coefficients. If ``non_linearity`` equals ``"topk"``,
            ``ths`` can indicate the number of wavelet coefficients
            that are kept (if ``int``) or the proportion of coefficients that are kept (if ``float``).
        :param float gamma: proximal operator stepsize.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        out = self.WaveletDenoiser(x, ths=ths * gamma)
        if self.clamp_min is not None:
            out = torch.clamp(out, min=self.clamp_min)
        if self.clamp_max is not None:
            out = torch.clamp(out, max=self.clamp_max)
        return out

    def psi(self, x, *args, **kwargs):
        r"""
        Applies the (flattening) wavelet decomposition of x.
        """
        return self.WaveletDenoiser.psi(
            x,
            wavelet=self.wv,
            level=self.level,
            dimension=self.wvdim,
            mode=self.mode,
            *args,
            **kwargs,
        )


class TVPrior(Prior):
    r"""
    Total variation (TV) prior :math:`\reg{x} = \| D x \|_{1,2}`.

    :param float def_crit: default convergence criterion for the inner solver of the TV denoiser; default value: 1e-8.
    :param int n_it_max: maximal number of iterations for the inner solver of the TV denoiser; default value: 1000.
    """

    def __init__(self, def_crit=1e-8, n_it_max=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.TVModel = TVDenoiser(crit=def_crit, n_it_max=n_it_max)

    def fn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the regularizer

        .. math::
            \reg{x} = \|Dx\|_{1,2}


        where D is the finite differences linear operator,
        and the 2-norm is taken on the dimension of the differences.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`g(x)`.
        """
        y = torch.sqrt(torch.sum(self.nabla(x) ** 2, dim=-1))
        return torch.sum(y.reshape(x.shape[0], -1), dim=-1)

    def prox(
        self, x: torch.Tensor, *args, gamma: float = 1.0, **kwargs
    ) -> torch.Tensor:
        r"""Compute the proximity operator of TV with the denoiser :class:`~deepinv.models.TVDenoiser`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        return self.TVModel(x, ths=gamma)

    def nabla(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.

        :param torch.Tensor x: the input tensor.
        :return: (:class:`torch.Tensor`) finite differences of x.
        """
        return self.TVModel.nabla(x)

    def nabla_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Applies the adjoint of the finite difference operator.

        :param torch.Tensor x: the input tensor.
        :return: (:class:`torch.Tensor`) adjoint of the finite differences of x.
        """
        return self.TVModel.nabla_adjoint(x)


class PatchPrior(Prior):
    r"""
    Patch prior :math:`g(x) = \sum_i h(P_i x)` for some prior :math:`h(x)` on the space of patches.

    Given a negative log likelihood (NLL) function on the patch space, this builds a prior by summing
    the NLLs of all (overlapping) patches in the image.

    :param Callable negative_patch_log_likelihood: NLL function on the patch space
    :param int n_patches: number of randomly selected patches for prior evaluation. -1 for taking all patches
    :param int patch_size: size of the patches
    :param bool | str pad: whether to use padding on the boundary to avoid undesired boundary effects. If `pad` is a string, it should be a valid padding mode for `torch.nn.functional.pad` (e.g. "reflect", "constant", etc.). If `pad` is `True`, the padding mode is set to "reflect". Default is `False`.
    """

    def __init__(
        self,
        negative_patch_log_likelihood: Callable,
        n_patches: int = -1,
        patch_size: int = 6,
        pad: bool | str = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.negative_patch_log_likelihood = negative_patch_log_likelihood
        self.explicit_prior = True
        self.n_patches = n_patches
        self.patch_size = patch_size

        if isinstance(pad, bool):
            self.pad = pad
            self.pad_mode = "reflect"
        elif isinstance(pad, str):
            if pad not in ["constant", "reflect", "replicate", "circular"]:
                raise ValueError(
                    f"Invalid padding mode {pad}. Should be one of 'constant', 'reflect', 'replicate' or 'circular'."
                )
            self.pad = True
            self.pad_mode = pad
        else:
            self.pad = False

    def fn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the regularizer

        .. math::
            \reg{x} = \sum_i h(P_i x)

        for some prior :math:`h(x)` on the space of patches, where :math:`P_i` is the operator extracting the :math:`i`-th patch from the image.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`g(x)`.
        """

        if self.pad:
            pad = self.patch_size - 1
            x = F.pad(x, (pad, pad, pad, pad), mode=self.pad_mode)

        patches, _ = patch_extractor(x, self.n_patches, self.patch_size)
        reg = self.negative_patch_log_likelihood(patches)
        reg = torch.mean(reg, -1)
        return reg


class GLOWCouplingBlock(nn.Module):
    r"""GLOW-style affine coupling block.

    Each block performs two successive affine coupling steps on the input vector,
    which is split into two halves :math:`(x_1, x_2)`:

    * Step 1 — a subnetwork acting on :math:`x_1` produces a pointwise scale :math:`s_1` and
      shift :math:`t_1` that are applied to :math:`x_2`: :math:`y_2 = x_2 \cdot \exp(s_1) + t_1`.
    * Step 2 — a second subnetwork acting on :math:`y_2` produces :math:`(s_2, t_2)` that are
      applied to :math:`x_1` : :math:`y_1 = x_1 \cdot \exp(s_2) + t_2`.

    Both steps are exactly invertible, and their combined log-determinant of the
    Jacobian is :math:`1^{\top}(s_1 + s_2)`.  The scale outputs are soft-clamped via
    :math:`\text{clamp} \times \frac{2}{\pi} \text{arctan}(s / \text{clamp})` to keep the log-determinant bounded and
    training stable.

    The two-step affine coupling structure follows :footcite:t:`dinh2017density`, and
    the soft-clamping of scales is introduced in :footcite:t:`kingma2018glow`.

    :param int dim: total input/output dimension (will be split evenly).
    :param Callable subnet: a callable ``subnet(channels_in, channels_out) -> nn.Module``
        that constructs the subnetworks used inside each coupling step.
    :param float clamp: soft-clamping magnitude for the log-scale outputs. Default is ``1.6``.

    """

    def __init__(self, dim: int, subnet: Callable, clamp: float = 1.6):
        super().__init__()
        self.clamp = clamp
        self.split1 = dim // 2
        self.split2 = dim - self.split1
        # subnet1: x1 -> (s1, t1) that acts on x2
        self.subnet1 = subnet(self.split1, self.split2 * 2)
        # subnet2: y2 -> (s2, t2) that acts on x1
        self.subnet2 = subnet(self.split2, self.split1 * 2)

    def _soft_clamp(self, s: torch.Tensor) -> torch.Tensor:
        r"""
        Applies soft clamping ``clamp * (2/π) * arctan(s / clamp)`` to the log-scale tensor.

        :param torch.Tensor s: unconstrained log-scale tensor.
        :return: (:class:`torch.Tensor`) soft-clamped log-scale tensor with values in ``(-clamp, clamp)``.
        """
        return self.clamp * (2.0 / np.pi) * torch.atan(s / self.clamp)

    def forward(self, x: torch.Tensor, rev: bool = False):
        r"""
        Applies the coupling block in the forward or inverse direction.

        :param torch.Tensor x: input tensor of shape ``(N, dim)``.
        :param bool rev: if ``True``, applies the inverse transformation. Default is ``False``.
        :return: tuple ``(y, log_det)`` where ``y`` (:class:`torch.Tensor`) is the
            transformed tensor of shape ``(N, dim)`` and ``log_det`` (:class:`torch.Tensor`)
            is the log-determinant of the Jacobian of shape ``(N,)``.
        """
        x1, x2 = x[:, : self.split1], x[:, self.split1 :]

        if not rev:
            # Step 1: transform x2 using x1
            st1 = self.subnet1(x1)
            s1 = self._soft_clamp(st1[:, : self.split2])
            t1 = st1[:, self.split2 :]
            y2 = x2 * torch.exp(s1) + t1
            log_det = s1.sum(dim=1)

            # Step 2: transform x1 using y2
            st2 = self.subnet2(y2)
            s2 = self._soft_clamp(st2[:, : self.split1])
            t2 = st2[:, self.split1 :]
            y1 = x1 * torch.exp(s2) + t2
            log_det = log_det + s2.sum(dim=1)

            return torch.cat([y1, y2], dim=1), log_det
        else:
            # Inverse step 2: recover x1 from x2 (which is y2 in forward)
            st2 = self.subnet2(x2)
            s2 = self._soft_clamp(st2[:, : self.split1])
            t2 = st2[:, self.split1 :]
            y1 = (x1 - t2) * torch.exp(-s2)
            log_det = -s2.sum(dim=1)

            # Inverse step 1: recover x2 using y1
            st1 = self.subnet1(y1)
            s1 = self._soft_clamp(st1[:, : self.split2])
            t1 = st1[:, self.split2 :]
            y2 = (x2 - t1) * torch.exp(-s1)
            log_det = log_det - s1.sum(dim=1)

            return torch.cat([y1, y2], dim=1), log_det


class NormalizingFlow(nn.Module):
    r"""Sequential normalizing flow built from GLOW-style affine coupling blocks.

    The flow maps an input sample ``x`` to a latent representation ``z`` by passing
    it through ``num_layers`` invertible coupling blocks in sequence.  The
    log-determinant of the full Jacobian is accumulated additively across blocks.
    Setting ``rev=True`` runs the blocks in reverse order to recover the original
    sample from a latent code.

    The architecture follows the generative flow of :footcite:t:`kingma2018glow`,
    using :class:`deepinv.optim.prior.GLOWCouplingBlock` as the building block.

    :param int dimension: dimension of each input sample (flattened patch size).
    :param int num_layers: number of coupling blocks to stack.
    :param Callable subnet: a callable ``subnet(channels_in, channels_out) -> nn.Module``
        that constructs the subnetworks used inside each coupling block.
    :param float clamp: soft-clamping magnitude passed to every coupling block. Default is ``1.6``.

    |sep|

    :Examples:


    >>> import torch
    >>> import torch.nn as nn
    >>> subnet = lambda c_in, c_out: nn.Sequential(
    ...     nn.Linear(c_in, 32), nn.ReLU(),
    ...     nn.Linear(32, 32), nn.ReLU(),
    ...     nn.Linear(32, c_out),
    ... )
    >>> flow = NormalizingFlow(dimension=8, num_layers=2, subnet=subnet)
    >>> x = torch.randn(4, 8)
    >>> z, log_det = flow(x)
    >>> z.shape
    torch.Size([4, 8])
    >>> log_det.shape
    torch.Size([4])
    >>> x_rec, _ = flow(z, rev=True)  # inverse flow recovers the input
    >>> torch.allclose(x, x_rec, atol=1e-5)
    True

    """

    def __init__(
        self, dimension: int, num_layers: int, subnet: Callable, clamp: float = 1.6
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GLOWCouplingBlock(dimension, subnet, clamp) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, rev: bool = False):
        r"""
        Passes the input through all coupling blocks sequentially.

        :param torch.Tensor x: input tensor of shape ``(N, dimension)``.
        :param bool rev: if ``True``, applies the blocks in reverse order (inverse flow). Default is ``False``.
        :return: tuple ``(z, log_det)`` where ``z`` (:class:`torch.Tensor`) is the latent
            representation of shape ``(N, dimension)`` and ``log_det`` (:class:`torch.Tensor`)
            is the total log-determinant of the Jacobian of shape ``(N,)``.
        """
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        blocks = self.blocks if not rev else reversed(self.blocks)
        for block in blocks:
            x, ld = block(x, rev=rev)
            log_det = log_det + ld
        return x, log_det


class PatchNR(Prior):
    r"""
    Patch prior via normalizing flows.

    The prior is defined as the sum of the negative log-likelihoods of all
    (overlapping) patches of the image under a learned normalizing flow model :footcite:p:`altekruger2023patchnr`.
    Denoting by :math:`P_i x` the :math:`i`-th patch of image :math:`x` (out of :math:`N`) and by
    :math:`f_\theta` the normalizing flow with parameters :math:`\theta`, the prior reads

    .. math::

        \reg{x} = \frac{1}{N} \sum_{i=1}^{N} -\log p_\theta(P_i x)

    where :math:`p_\theta` is the patch distribution implicitly defined by the flow.
    Applying the change-of-variables formula with the standard Gaussian base distribution
    :math:`p_z = \mathcal{N}(0, I)`, this expands to

    .. math::

        \reg{x} = \frac{1}{N} \sum_{i=1}^{N} \left(
            \frac{1}{2}\| f_\theta(P_i x) \|_2^2
            - \log \left|\det J_{f_\theta}(P_i x)\right|
        \right)

    where :math:`J_{f_\theta}` is the Jacobian of :math:`f_\theta`.  Both terms are
    computed in a single forward pass through the flow, which returns the latent code
    :math:`z = f_\theta(P_i x)` and the log-determinant
    :math:`\log |\det J_{f_\theta}(P_i x)|` simultaneously.

    The forward method evaluates this negative log likelihood.

    :param torch.nn.Module normalizing_flow: describes the normalizing flow of the model. Generally it can be any :class:`torch.nn.Module`
        supporting backpropagation. It takes a (batched) tensor of flattened patches and the boolean ``rev`` (default ``False``)
        as input and returns ``(latent, log_det_jacobian)`` as output.
        If ``rev=True``, it applies the inverse of the flow.
        When set to ``None``, a GLOW-style invertible neural network is built, where the number of
        coupling blocks and the hidden neurons of the sub-networks are determined by ``num_layers`` and ``sub_net_size`` respectively.
        If `None`, it is set to :class:`deepinv.optim.prior.NormalizingFlow` model.
    :param str pretrained: Define pretrained weights by its path checkpoint, `None` for random initialization,
        `"PatchNR_lodopab_small2"` for the weights from the :ref:`limited-angle CT example <sphx_glr_auto_examples_optimization_demo_patch_priors_CT.py>`.
    :param int patch_size: size of patches
    :param int channels: number of channels for the underlying images/patches.
    :param int num_layers: defines the number of coupling blocks of the normalizing flow if `normalizing_flow` is ``None``.
    :param int sub_net_size: defines the number of hidden neurons in the subnetworks of the normalizing flow
        if `normalizing_flow` is ``None``.
    :param str device: used device

    |sep|

    :Examples:

    >>> import torch
    >>> import deepinv as dinv
    >>> prior = dinv.optim.PatchNR(patch_size=6, channels=1)
    >>> x = torch.randn(2, 10, 36)  # (batch, n_patches, patch_size^2 * channels)
    >>> nll = prior.fn(x)
    >>> nll.shape
    torch.Size([2, 10])

    """

    def __init__(
        self,
        normalizing_flow: nn.Module = None,
        pretrained: str = None,
        patch_size: int = 6,
        channels: int = 1,
        num_layers: int = 5,
        sub_net_size: int = 256,
        device="cpu",
    ):
        super(PatchNR, self).__init__()
        if normalizing_flow is None:
            dimension = patch_size**2 * channels

            def subnet_fc(c_in, c_out):
                return nn.Sequential(
                    nn.Linear(c_in, sub_net_size),
                    nn.ReLU(),
                    nn.Linear(sub_net_size, sub_net_size),
                    nn.ReLU(),
                    nn.Linear(sub_net_size, c_out),
                )

            self.normalizing_flow = NormalizingFlow(
                dimension=dimension,
                num_layers=num_layers,
                subnet=subnet_fc,
                clamp=1.6,
            ).to(device)
        else:
            self.normalizing_flow = normalizing_flow
        if pretrained:
            if pretrained == "PatchNR_lodopab_small2":
                if patch_size != 3:  # pragma: no cover
                    raise ValueError(
                        f"PatchNR_lodopab_small requires patch_size 3, but got {patch_size}"
                    )
                if channels != 1:  # pragma: no cover
                    raise ValueError(
                        f"PatchNR_lodopab_small requires channels 1, but got {channels}"
                    )
                file_name = "PatchNR_lodopab_small2.pt"
                url = get_weights_url(model_name="demo", file_name=file_name)
                weights = load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=file_name
                )
            else:
                weights = torch.load(pretrained, map_location=device)

            self.load_state_dict(weights)

    def fn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Evaluates the negative log likelihood function of th PatchNR.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`g(x)`.
        """
        B, n_patches = x.shape[0:2]
        latent_x, logdet = self.normalizing_flow(x.view(B * n_patches, -1))
        logpz = 0.5 * torch.sum(latent_x.view(B, n_patches, -1) ** 2, -1)
        return logpz - logdet.view(B, n_patches)


class L12Prior(Prior):
    r"""
    :math:`\ell_{1,2}` prior :math:`\reg{x} = \sum_i\| x_i \|_2`.

    The :math:`\ell_2` norm is computed over a tensor axis that can be defined by the user. By default, ``l2_axis=-1``.

    :param int l2_axis: dimension in which the :math:`\ell_2` norm is computed.

    |sep|

    :Examples:

    >>> import torch
    >>> from deepinv.optim import L12Prior
    >>> seed = torch.manual_seed(0) # Random seed for reproducibility
    >>> x = torch.randn(2, 1, 3, 3) # Define random 3x3 image
    >>> prior = L12Prior()
    >>> prior.fn(x)
    tensor([5.4949, 4.3881])
    >>> prior.prox(x)
    tensor([[[[-0.4666, -0.4776,  0.2348],
              [ 0.3636,  0.2744, -0.7125],
              [-0.1655,  0.8986,  0.2270]]],
    <BLANKLINE>
    <BLANKLINE>
            [[[-0.0000, -0.0000,  0.0000],
              [ 0.7883,  0.9000,  0.5369],
              [-0.3695,  0.4081,  0.5513]]]])

    """

    def __init__(self, *args, l2_axis: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.l2_axis = l2_axis

    def fn(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Computes the regularizer :math:`\reg{x} = \sum_i\| x_i \|_2`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        x = torch.linalg.vector_norm(x, dim=self.l2_axis, ord=2, keepdim=False)
        return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=-1, ord=1)

    def prox(
        self, x: torch.Tensor, *args, gamma: float = 1.0, **kwargs
    ) -> torch.Tensor:
        r"""
        Calculates the proximity operator of the :math:`\ell_{1,2}` function at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = (1 - \frac{\gamma}{\mathrm{max}(\Vert x \Vert_2,\gamma)}) x


        where :math:`\gamma` is a stepsize.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param int l2_axis: axis in which the l2 norm is computed.
        :return torch.Tensor: proximity operator at :math:`x`.
        """

        z = torch.linalg.vector_norm(
            x, dim=self.l2_axis, ord=2, keepdim=True
        )  # Compute the norm
        # 1 - gamma/max(z, gamma) = relu(z - gamma) / z, adding 1e-12 to avoid division by 0
        z = torch.nn.functional.relu(z - gamma) / (z + 1e-12)
        return z * x
