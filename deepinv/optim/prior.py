import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from deepinv.optim.potential import Potential
from deepinv.models.tv import TVDenoiser
from deepinv.models.wavdict import WaveletDenoiser, WaveletDictDenoiser
from deepinv.utils import patch_extractor
from deepinv.models.GSPnP import GSDRUNet
from deepinv.optim.utils import nonmonotone_accelerated_proximal_gradient


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

    def __init__(self, g=None):
        super().__init__(fn=g)
        self.explicit_prior = False if self._fn is None else True


class Zero(Prior):
    r"""
    Zero prior :math:`\reg{x} = 0`.
    """

    def __init__(self):
        super().__init__()

        def forward(x, *args, **kwargs):
            return torch.tensor(0.0)

        self._g = forward
        self.explicit_prior = True

    def fn(self, x, *args, **kwargs):
        r"""
        Computes the zero prior :math:`\reg(x) = 0` at :math:`x`.

        It returns a tensor of zeros of the same shape as :math:`x`.
        """
        return torch.zeros_like(x)

    def grad(self, x, *args, **kwargs):
        r"""
        Computes the gradient of the zero prior :math:`\reg(x) = 0` at :math:`x`.

        It returns a tensor of zeros of the same shape as :math:`x`.
        """
        return torch.zeros_like(x)

    def prox(self, x, ths=1.0, gamma=1.0, *args, **kwargs):
        r"""
        Computes the proximal operator of the zero prior :math:`\reg(x) = 0` at :math:`x`.

        It returns the identity :math:`x`.
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

    def prox(self, x, sigma_denoiser, *args, **kwargs):
        r"""
        Uses denoising as the proximity operator of the PnP prior :math:`\regname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float sigma_denoiser: noise level parameter of the denoiser.
        :return: (torch.tensor) proximity operator at :math:`x`.
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

    def grad(self, x, sigma_denoiser, *args, **kwargs):
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

    def grad(self, x, sigma_denoiser, *args, **kwargs):
        r"""
        Applies the denoiser to the input signal.

        :param torch.Tensor x: the input tensor.
        :param float sigma_denoiser: the noise level.
        """
        return self.stable_division(
            x - self.denoiser(x, sigma_denoiser, *args, **kwargs), sigma_denoiser**2
        )

    def score(self, x, sigma_denoiser, *args, **kwargs):
        r"""
        Computes the score function :math:`\nabla \log p_\sigma`, using Tweedie's formula.

        :param torch.Tensor x: the input tensor.
        :param float sigma_denoiser: the noise level.
        """
        return self.stable_division(
            self.denoiser(x, sigma_denoiser, *args, **kwargs) - x, sigma_denoiser**2
        )

    @staticmethod
    def stable_division(a, b, epsilon: float = 1e-7):
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

    def fn(self, x, *args, **kwargs):
        r"""
        Computes the Tikhonov regularizer :math:`\reg{x} = \frac{1}{2}\| x \|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        return 0.5 * torch.norm(x.contiguous().view(x.shape[0], -1), p=2, dim=-1) ** 2

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the Tikhonov regularization term :math:`\regname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (:class:`torch.Tensor`) gradient at :math:`x`.
        """
        return x

    def prox(self, x, *args, gamma=1.0, **kwargs):
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

    def fn(self, x, *args, **kwargs):
        r"""
        Computes the regularizer :math:`\reg{x} = \| x \|_1`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        return torch.norm(x.contiguous().view(x.shape[0], -1), p=1, dim=-1)

    def prox(self, x, *args, ths=1.0, gamma=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the l1 regularization term :math:`\regname` at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = \operatorname{sign}(x) \max(|x| - \gamma, 0)


        where :math:`\gamma` is a stepsize.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return torch.Tensor: proximity operator at :math:`x`.
        """
        return (x.abs() - ths * gamma).clamp(min=0.0) * x.sign()


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
    :param str mode: padding mode for the wavelet transform (default: "zero").
    :param float clamp_min: minimum value for the clamping. Default is None.
    :param float clamp_max: maximum value for the clamping. Default is None.
    """

    def __init__(
        self,
        level=3,
        wv="db8",
        p=1,
        device="cpu",
        wvdim=2,
        mode="zero",
        clamp_min=None,
        clamp_max=None,
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
                wvdim=self.wvdim,
            )
        elif type(self.wv) == list:
            self.WaveletDenoiser = WaveletDictDenoiser(
                level=self.level,
                list_wv=self.wv,
                max_iter=10,
                non_linearity=self.non_linearity,
                wvdim=self.wvdim,
            )
        else:
            raise ValueError(
                f"wv should be a string (name of the wavelet) or a list of strings (list of wavelet names). Got {type(self.wv)} instead."
            )

    def fn(self, x, *args, reduce=True, **kwargs):
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

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""Compute the proximity operator of the wavelet prior with the denoiser :class:`~deepinv.models.WaveletDenoiser`.
        Only detail coefficients are thresholded.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        out = self.WaveletDenoiser(x, ths=gamma)
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

    def fn(self, x, *args, **kwargs):
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

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""Compute the proximity operator of TV with the denoiser :class:`~deepinv.models.TVDenoiser`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (:class:`torch.Tensor`) proximity operator at :math:`x`.
        """
        return self.TVModel(x, ths=gamma)

    def nabla(self, x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        return self.TVModel.nabla(x)

    def nabla_adjoint(self, x):
        r"""
        Applies the adjoint of the finite difference operator.
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
    :param bool pad: whether to use mirror padding on the boundary to avoid undesired boundary effects
    """

    def __init__(
        self,
        negative_patch_log_likelihood,
        n_patches=-1,
        patch_size=6,
        pad=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.negative_patch_log_likelihood = negative_patch_log_likelihood
        self.explicit_prior = True
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pad = pad

    def fn(self, x, *args, **kwargs):
        if self.pad:
            x = torch.cat(
                (
                    torch.flip(x[:, :, -self.patch_size : -1, :], (2,)),
                    x,
                    torch.flip(x[:, :, 1 : self.patch_size, :], (2,)),
                ),
                2,
            )
            x = torch.cat(
                (
                    torch.flip(x[:, :, :, -self.patch_size : -1], (3,)),
                    x,
                    torch.flip(x[:, :, :, 1 : self.patch_size], (3,)),
                ),
                3,
            )

        patches, _ = patch_extractor(x, self.n_patches, self.patch_size)
        reg = self.negative_patch_log_likelihood(patches)
        reg = torch.mean(reg, -1)
        return reg


class PatchNR(Prior):
    r"""
    Patch prior via normalizing flows.

    The forward method evaluates its negative log likelihood.

    :param torch.nn.Module normalizing_flow: describes the normalizing flow of the model. Generally it can be any :class:`torch.nn.Module`
        supporting backpropagation. It takes a (batched) tensor of flattened patches and the boolean rev (default `False`)
        as input and provides the value and the log-determinant of the Jacobian of the normalizing flow as an output
        If `rev=True`, it considers the inverse of the normalizing flow.
        When set to ``None`` it is set to a dense invertible neural network built with the FrEIA library, where the number of
        invertible blocks and the size of the subnetworks is determined by the parameters `num_layers` and `sub_net_size`.
    :param str pretrained: Define pretrained weights by its path to a `.pt` file, None for random initialization,
        `"PatchNR_lodopab_small"` for the weights from the limited-angle CT example.
    :param int patch_size: size of patches
    :param int channels: number of channels for the underlying images/patches.
    :param int num_layers: defines the number of blocks of the generated normalizing flow if `normalizing_flow` is ``None``.
    :param int sub_net_size: defines the number of hidden neurons in the subnetworks of the generated normalizing flow
        if `normalizing_flow` is ``None``.
    :param str device: used device

    .. note::

        This class requires the ``FrEIA`` package to be installed. Install with ``pip install FrEIA``.
    """

    def __init__(
        self,
        normalizing_flow=None,
        pretrained=None,
        patch_size=6,
        channels=1,
        num_layers=5,
        sub_net_size=256,
        device="cpu",
    ):
        import FrEIA.framework as Ff
        import FrEIA.modules as Fm

        super(PatchNR, self).__init__()
        if normalizing_flow is None:
            # Create Normalizing Flow with FrEIA
            dimension = patch_size**2 * channels

            def subnet_fc(c_in, c_out):
                return nn.Sequential(
                    nn.Linear(c_in, sub_net_size),
                    nn.ReLU(),
                    nn.Linear(sub_net_size, sub_net_size),
                    nn.ReLU(),
                    nn.Linear(sub_net_size, c_out),
                )

            nodes = [Ff.InputNode(dimension, name="input")]
            for k in range(num_layers):
                nodes.append(
                    Ff.Node(
                        nodes[-1],
                        Fm.GLOWCouplingBlock,
                        {"subnet_constructor": subnet_fc, "clamp": 1.6},
                        name=f"coupling_{k}",
                    )
                )
            nodes.append(Ff.OutputNode(nodes[-1], name="output"))

            self.normalizing_flow = Ff.GraphINN(nodes, verbose=False).to(device)
        else:
            self.normalizing_flow = normalizing_flow
        if pretrained:
            if pretrained[-3:] == ".pt":
                weights = torch.load(pretrained, map_location=device)
            else:
                if pretrained.startswith("PatchNR_lodopab_small"):
                    assert patch_size == 3
                    assert channels == 1
                    file_name = "PatchNR_lodopab_small.pt"
                    url = "https://drive.google.com/uc?export=download&id=1Z2us9ZHjDGOlU6r1Jee0s2BBej2XV5-i"
                else:
                    raise ValueError("Pretrained weights not found!")
                weights = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=file_name
                )
            self.normalizing_flow.load_state_dict(weights)

    def fn(self, x, *args, **kwargs):
        r"""
        Evaluates the negative log likelihood function of th PatchNR.

        :param torch.Tensor x: image tensor
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

    def __init__(self, *args, l2_axis=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.l2_axis = l2_axis

    def fn(self, x, *args, **kwargs):
        r"""
        Computes the regularizer :math:`\reg{x} = \sum_i\| x_i \|_2`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        x_l2 = torch.norm(x, p=2, dim=self.l2_axis)
        return torch.norm(x_l2.reshape(x.shape[0], -1), p=1, dim=-1)

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the :math:`\ell_{1,2}` function at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = (1 - \frac{\gamma}{max{\Vert x \Vert_2,\gamma}}) x


        where :math:`\gamma` is a stepsize.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param int l2_axis: axis in which the l2 norm is computed.
        :return torch.Tensor: proximity operator at :math:`x`.
        """

        z = torch.norm(x, p=2, dim=self.l2_axis, keepdim=True)  # Compute the norm
        z2 = torch.max(
            z, gamma * torch.ones_like(z)
        )  # Compute its max w.r.t. gamma at each point
        z3 = torch.ones_like(z)  # Construct a mask of ones
        mask_z = z > 0  # Find locations where z (hence x) is not already zero
        z3[mask_z] = (
            z3[mask_z] - gamma / z2[mask_z]
        )  # If z < gamma -> z2 = gamma -> z3 -gamma/gamma =0  (threshold below gamma)
        # Oth. z3 = 1- gamma/z2
        z4 = torch.multiply(
            x, z3
        )  # All elems of x with norm < gamma are set 0; the others are z4 = x(1-gamma/|x|)
        # Creating a mask to avoid diving by zero
        # if an element of z is zero, then it is zero in x, therefore torch.multiply(z, x) is zero as well
        return z4


class WCRR(Prior):
    r"""
    (Weakly) Convex Ridge Regularizer :math:`\reg{x}=\sum_{c} \psi_c(W_c x)`

    for filters :math:`W_c` and potentials :math:`\psi_c`. The filters :math:`W_c` are realized by a concatination multiple convolution
    layers without nonlinearity. The potentials :math:`\psi_c` are given by scaled versions smoothed absolute values,
    see :footcite:t:`hertrich2025learning` for a precise description.

    To allow the automatic tuning of the regularization parameter, we parameterize the regularizer with two additional scalings, i.e.,
    we implement :math:`\frac{\alpha}{\sigma^2}\reg{\sigma x}` instead of :math:`\reg{x}` where :math:`\alpha` and :math:`\sigma` are learnable parameters of the regularizer.
    If the weak CRR is used, :math:`\alpha` is fixed per default, since it changes the weak convexity constant.

    The (W)CRR was introduced by :footcite:t:`goujon2023neural` and :footcite:t:`goujon2024learning`.
    The specific implementation is taken from :footcite:t:`hertrich2025learning`.

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
        in_channels=3,
        weak_convexity=0.0,
        nb_channels=(4, 8, 64),
        filter_sizes=(5, 5, 5),
        device="cpu",
        pretrained="download",
        warn_output_scaling=True,
    ):
        super(WCRR, self).__init__()
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
        # output scaling is not trainable for weak_convexity > 0 (to preserve the weak convexity)
        self.output_scaling = nn.Parameter(
            torch.tensor(0.0, device=device)
        ).requires_grad_(weak_convexity == 0.0)
        self.weak_cvx = weak_convexity

        if pretrained is not None:
            if pretrained == "download":
                if in_channels == 1 and weak_convexity == 0.0:
                    file_name = "CRR_gray.pt"
                    url = "https://drive.google.com/uc?export=download&id=1Yz2eSCM85EaGQTDviPnmqMY1ySqti3hr"
                elif in_channels == 3 and weak_convexity == 0.0:
                    file_name = "CRR_color.pt"
                    url = "https://drive.google.com/uc?export=download&id=1MBXxuHGmRBEalMOE4fNuCHpiIp3yFo4J"
                elif in_channels == 1 and weak_convexity == 1.0:
                    file_name = "WCRR_gray.pt"
                    url = "https://drive.google.com/uc?export=download&id=10Gg_C0EE-ItWCxEPDSriRz-CICL9ythY"
                elif in_channels == 3 and weak_convexity == 1.0:
                    file_name = "WCRR_color.pt"
                    url = "https://drive.google.com/uc?export=download&id=1Z6LW7utP8xTTvb8jktugT-E-wOM4KX_h"
                else:
                    raise ValueError(
                        "Weights are only available for weak_convexity in [0.0, 1.0] and in_channels in [1, 3]!"
                    )
                weights = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=file_name
                )
                self.load_state_dict(weights, strict=True)
            else:
                self.load_state_dict(torch.load(pretrained, map_location=device))

    def smooth_l1(self, x):
        return torch.clip(x**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(x), 1.0) - 1.0

    def grad_smooth_l1(self, x):
        return torch.clip(x, -1.0, 1.0)

    def get_conv_lip(self):
        impulse = self.filters(self.dirac)
        for filt in reversed(self.filters):
            impulse = F.conv_transpose2d(impulse, filt.weight, padding=filt.padding)
        return torch.fft.fft2(impulse, s=[256, 256]).abs().max()

    def conv(self, x):
        x = x / torch.sqrt(self.get_conv_lip())
        return self.filters(x)

    def conv_transpose(self, x):
        x = x / torch.sqrt(self.get_conv_lip())
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
        grad = self.conv(x)
        grad = grad * torch.exp(self.scaling + self.input_scaling)
        if get_energy:
            reg = (
                self.smooth_l1(torch.exp(self.beta) * grad) * torch.exp(-self.beta)
                - self.smooth_l1(grad) * self.weak_cvx
            )
            reg = reg * torch.exp(
                self.output_scaling - 2 * self.scaling - 2 * self.input_scaling
            )
            reg = reg.sum(dim=(1, 2, 3))
        grad = (
            self.grad_smooth_l1(torch.exp(self.beta) * grad)
            - self.grad_smooth_l1(grad) * self.weak_cvx
        )
        grad = grad * torch.exp(self.output_scaling - self.scaling - self.input_scaling)
        grad = self.conv_transpose(grad)
        if get_energy:
            return reg, grad
        return grad

    def fn(self, x, *args, **kwargs):
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
                "The parameter WCRR.output_scaling is not zero even though WCRR.weak_convexity is not zero! "
                + "This means that the weak convexity parameter of the WCRR is not WCRR.weak_convexity but exp(output_scaling)*WCRR.weak_convexity. "
                + "If you require the WCRR to keep the weak convexity, set WCRR.output_scaling.requires_grad_(False) for all training methods and do not "
                + "change WCRR.output_scaling. To suppress this warning, set warn_output_scaling in the constructor of the WCRR to False."
            )
        reg = self.conv(x)
        reg = reg * torch.exp(self.scaling + self.input_scaling)
        reg = (
            self.smooth_l1(torch.exp(self.beta) * reg) * torch.exp(-self.beta)
            - self.smooth_l1(reg) * self.weak_cvx
        )
        reg = reg * torch.exp(
            self.output_scaling - 2 * self.scaling - 2 * self.input_scaling
        )
        reg = reg.sum(dim=(1, 2, 3))
        return reg

    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        return super()._apply(fn)

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the the regularizer at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = \argmin_z \frac{1}{2}\|z-x\|^2 + \gamma g(x)


        where :math:`\gamma` is a stepsize. The minimizer is computed using the
        :class:`nonmonotonic accelerated (proximal) gradient <deepinv.optim.NMAPG>` algorithm.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param int l2_axis: axis in which the l2 norm is computed.
        :return torch.Tensor: proximity operator at :math:`x`.
        """
        f = lambda z, y: 0.5 * torch.sum((z - y) ** 2, (1, 2, 3)) + gamma * self(z)
        nabla_f = lambda z, y: z - y + gamma * self.grad(z)

        def f_and_nabla(z, y):
            with torch.no_grad():
                out_f, out_grad = self.grad(z, get_energy=True)
            return (
                0.5 * torch.sum((z - y) ** 2, (1, 2, 3)) + out_f.detach(),
                z - y + out_grad.detach(),
            )

        return nonmonotone_accelerated_proximal_gradient(
            x, f, nabla_f=nabla_f, f_and_nabla=f_and_nabla, y=x
        )[0]


class LSR(Prior):
    r"""
    Least Squares Regularizer :math:`\reg{x}=\|x-D(x)\|^2` for a DRUNet :math:`D`.

    To allow the automatic tuning of the regularization parameter, we parameterize the regularizer with two additional scalings, i.e.,
    we implement :math:`\alpha\reg{\sigma x}` instead of :math:`\reg{x}` where :math:`\alpha` and :math:`\sigma` are learnable parameters of the regularizer.
    These parameters are learned in the log scale to enforce positivity.

    This type of network was used in several references, see e.g., :footcite:t:`hurault2021gradient` or :footcite:t:`zou2023deep`.
    The specific implementation wraps the :class:`GSDRUNet<deepinv.models.GSPnP.GSDRUNet>`.

    :param int in_channels: Number of input channels (`1` for gray valued images, `3` for color images). Default: `3`
    :param str device: Device for the weights. Default: `"cpu"`
    :param str, None pretrained: use pretrained weights. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with 3 or 1 input/output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-learned-reg>` for more details.
    :param list of int nc: number of channels of the DRUNet, cf. :class:`deepinv.models.DRUNet`. Default: `[32, 64, 128, 256]`
    :param int nb: number of residual blocks of the DRUNet, cf. :class:`deepinv.models.DRUNet`. Default: `2`.
    :param deepinv.models.GSPnP.GSDRUNet pretrained_GSDRUNet: If already a GSDRUNet object exists, a LSR with this GSDRUNet can be created
        by passing it through this argument. `None` for initializing a new GSDRUNet. Default: `None`
    :param float alpha: scaling factor in the GSDRUNet. Default: `1.0`
    :param float sigma: Noise level applied in the DRUNet. Default: `0.03`
    """

    def __init__(
        self,
        in_channels=3,
        device="cpu",
        pretrained="download",
        nc=(32, 64, 128, 256),
        nb=2,
        pretrained_GSDRUNet=None,
        alpha=1.0,
        sigma=0.03,
    ):
        super(LSR, self).__init__()

        if pretrained_GSDRUNet is None:
            self.model = GSDRUNet(
                alpha=alpha,
                in_channels=in_channels,
                out_channels=in_channels,
                nb=nb,
                nc=nc,
                act_mode="s",
                pretrained=None,
                device=device,
            )
        elif isinstance(pretrained_GSDRUNet, GSDRUNet):
            self.model = pretrained_GSDRUNet.to(device)
            self.add_module("model", self.model)
        else:
            raise ValueError(
                "The parameter pretrained_GSDRUNet must either be None or an instance of GSDRUNet!"
            )

        self.model.detach = False

        self.input_scaling = nn.Parameter(torch.tensor(0.0, device=device))
        self.output_scaling = nn.Parameter(torch.tensor(0.0, device=device))

        self.sigma = sigma

        if pretrained is not None:
            if pretrained == "download":
                if in_channels == 1:
                    file_name = "LSR_gray.pt"
                    url = "https://drive.google.com/uc?export=download&id=1YclYsQe7eM7l9Cmp9bxqT7f2WbOF0SKy"
                elif in_channels == 3:
                    file_name = "LSR_color.pt"
                    url = "https://drive.google.com/uc?export=download&id=1am3EG6XQubZM3oO08ByKyC2sRPrc7VzV"
                else:
                    raise ValueError(
                        "Weights are only available for in_channels in [1, 3]!"
                    )
                weights = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=file_name
                )
                self.load_state_dict(weights, strict=True)
            else:
                self.load_state_dict(torch.load(pretrained, map_location=device))

    def grad(self, x, *args, get_energy=False, **kwargs):
        r"""
        Calculates the gradient of the regularizer at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param bool get_energy: Optional flag. If set to True, the function additionally returns the objective value at :math:`x`. Dafault: False.
        :return: (:class:`torch.Tensor`) gradient at :math:`x`.
        """
        grad = torch.exp(self.output_scaling) * self.model.potential_grad(
            torch.exp(self.input_scaling) * x.contiguous(), self.sigma
        )
        if get_energy:
            reg = self(x)
            return reg, grad
        return grad

    def fn(self, x, *args, **kwargs):
        r"""
        Computes the regularizer :math:`\reg{x}=\|x-D(x)\|^2`

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (:class:`torch.Tensor`) prior :math:`\reg{x}`.
        """
        return torch.exp(
            self.output_scaling + self.input_scaling
        ) * self.model.potential(
            torch.exp(self.input_scaling) * x.contiguous(), self.sigma
        )

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the the regularizer at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma g}(x) = \argmin_z \frac{1}{2}\|z-x\|^2 + \gamma g(x)


        where :math:`\gamma` is a stepsize. The minimizer is computed using the
        :class:`nonmonotonic accelerated (proximal) gradient <deepinv.optim.NMAPG>` algorithm.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param int l2_axis: axis in which the l2 norm is computed.
        :return torch.Tensor: proximity operator at :math:`x`.
        """
        f = lambda z, y: 0.5 * torch.sum((z - y) ** 2, (1, 2, 3)) + gamma * self(z)
        nabla_f = lambda z, y: z - y + gamma * self.grad(z)

        def f_and_nabla(z, y):
            with torch.no_grad():
                out_f, out_grad = self.grad(z, get_energy=True)
            return (
                0.5 * torch.sum((z - y) ** 2, (1, 2, 3)) + out_f.detach(),
                z - y + out_grad.detach(),
            )

        return nonmonotone_accelerated_proximal_gradient(
            x, f, nabla_f=nabla_f, f_and_nabla=f_and_nabla, y=x
        )[0]
