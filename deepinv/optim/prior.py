import numpy as np

import torch
import torch.nn as nn

try:
    import FrEIA.framework as Ff
    import FrEIA.modules as Fm
except:
    Ff = ImportError("The FrEIA package is not installed.")
    Fm = ImportError("The FrEIA package is not installed.")

from deepinv.optim.utils import gradient_descent
from deepinv.models.tv import TVDenoiser
from deepinv.models.wavdict import WaveletDenoiser
from deepinv.utils import patch_extractor


class Prior(nn.Module):
    r"""
    Prior term :math:`\reg{x}`.

    This is the base class for the prior term :math:`\reg{x}`. Similarly to the :meth:`deepinv.optim.DataFidelity` class,
    this class comes with methods for computing
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


    :param callable g: Prior function :math:`g(x)`.
    """

    def __init__(self, g=None):
        super().__init__()
        self._g = g
        self.explicit_prior = False if self._g is None else True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self._g(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        r"""
        Computes the prior :math:`g(x)`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.tensor) prior :math:`g(x)`.
        """
        return self.g(x, *args, **kwargs)

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the prior term :math:`\regname` at :math:`x`.
        By default, the gradient is computed using automatic differentiation.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.tensor) gradient :math:`\nabla_x g`, computed in :math:`x`.
        """
        with torch.enable_grad():
            x = x.requires_grad_()
            grad = torch.autograd.grad(
                self.g(x, *args, **kwargs), x, create_graph=True, only_inputs=True
            )[0]
        return grad

    def prox(
        self,
        x,
        *args,
        gamma=1.0,
        stepsize_inter=1.0,
        max_iter_inter=50,
        tol_inter=1e-3,
        **kwargs,
    ):
        r"""
        Calculates the proximity operator of :math:`\regname` at :math:`x`. By default, the proximity operator is computed using internal gradient descent.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float stepsize_inter: stepsize used for internal gradient descent
        :param int max_iter_inter: maximal number of iterations for internal gradient descent.
        :param float tol_inter: internal gradient descent has converged when the L2 distance between two consecutive iterates is smaller than tol_inter.
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma g}(x)`, computed in :math:`x`.
        """
        grad = lambda z: gamma * self.grad(z, *args, **kwargs) + (z - x)
        return gradient_descent(
            grad, x, step_size=stepsize_inter, max_iter=max_iter_inter, tol=tol_inter
        )

    def prox_conjugate(self, x, *args, gamma=1.0, lamb=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the convex conjugate :math:`(\lambda g)^*` at :math:`x`, using the Moreau formula.

        ::Warning:: Only valid for convex :math:`\regname`

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float lamb: math:`\lambda` parameter in front of :math:`f`
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma \lambda g)^*}(x)`, computed in :math:`x`.
        """
        return x - gamma * self.prox(x / gamma, lamb / gamma, *args, **kwargs)


class Zero(Prior):
    r"""
    Zero prior :math:`\reg{x} = 0`.
    """

    def __init__(self):
        super().__init__()
        self._g = lambda x: 0.0
        self.explicit_prior = True

    def grad(self, x, *args):
        r"""
        Computes the gradient of the zero prior :math:`\reg(x) = 0` at :math:`x`.

        It returns a tensor of zeros of the same shape as :math:`x`.
        """
        return torch.zeros_like(x)

    def prox(self, x, ths=1.0, gamma=1.0):
        r"""
        Computes the proximal operator of the zero prior :math:`\reg(x) = 0` at :math:`x`.

        It returns the identity :math:`x`.
        """
        return x


class PnP(Prior):
    r"""
    Plug-and-play prior :math:`\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x)`.


    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
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


    :param callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
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

        If math:`\sigma=1`, this prior is equal to :class:`deepinv.optim.RED`, which is defined in
        `Regularization by Denoising (RED) <https://arxiv.org/abs/1611.02862>`_ and doesn't require the normalization.


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

    def grad(self, x, sigma_denoiser):
        r"""
        Applies the denoiser to the input signal.

        :param torch.Tensor x: the input tensor.
        :param float sigma_denoiser: the noise level.
        """
        return (1 / sigma_denoiser**2) * (x - self.denoiser(x, sigma_denoiser))


class Tikhonov(Prior):
    r"""
    Tikhonov regularizer :math:`\reg{x} = \frac{1}{2}\| x \|_2^2`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the Tikhonov regularizer :math:`\reg{x} = \frac{1}{2}\| x \|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`\reg{x}`.
        """
        return 0.5 * torch.norm(x.contiguous().view(x.shape[0], -1), p=2, dim=-1) ** 2

    def grad(self, x, *args, **kwargs):
        r"""
        Calculates the gradient of the Tikhonov regularization term :math:`\regname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.Tensor) gradient at :math:`x`.
        """
        return x

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""
        Calculates the proximity operator of the Tikhonov regularization term :math:`\gamma g` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """
        return (1 / (gamma + 1)) * x


class L1Prior(Prior):
    r"""
    :math:`\ell_1` prior :math:`\reg{x} = \| x \|_1`.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, *args, **kwargs):
        r"""
        Computes the regularizer :math:`\reg{x} = \| x \|_1`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`\reg{x}`.
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
        return torch.sign(x) * torch.max(
            torch.abs(x) - ths * gamma, torch.zeros_like(x)
        )


class WaveletPrior(Prior):
    r"""
    Wavelet prior :math:`\reg{x} = \|\Psi x\|_{p}`.

    :math:`\Psi` is an orthonormal wavelet transform, and :math:`\|\cdot\|_{p}` is the :math:`p`-norm, with
    :math:`p=0`, :math:`p=1`, or :math:`p=\infty`.

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
    """

    def __init__(self, level=3, wv="db8", p=1, device="cpu", wvdim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.p = p
        self.wv = wv
        self.wvdim = wvdim
        self.level = level
        self.device = device
        if p == 0:
            self.non_linearity = "hard"
        elif p == 1:
            self.non_linearity = "soft"
        elif p == np.inf or p == "inf":
            self.non_linearity = "topk"
        else:
            raise ValueError("p should be 0, 1 or inf")
        self.WaveletDenoiser = WaveletDenoiser(
            level=self.level,
            wv=self.wv,
            device=self.device,
            non_linearity=self.non_linearity,
            wvdim=self.wvdim,
        )

    def g(self, x, *args, reduce=True, **kwargs):
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
        :return: (torch.Tensor) prior :math:`g(x)`.
        """
        list_dec = self.psi(x)
        list_norm = torch.hstack([torch.norm(dec, p=self.p) for dec in list_dec])
        if reduce:
            return torch.sum(list_norm)
        else:
            return list_norm

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""Compute the proximity operator of the wavelet prior with the denoiser :class:`~deepinv.models.WaveletDenoiser`.
        Only detail coefficients are thresholded.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """
        return self.WaveletDenoiser(x, ths=gamma)

    def psi(self, x):
        r"""
        Applies the (flattening) wavelet decomposition of x.
        """
        return self.WaveletDenoiser.psi(x, self.wv, self.level, self.wvdim)


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

    def g(self, x, *args, **kwargs):
        r"""
        Computes the regularizer

        .. math::
            g(x) = \|Dx\|_{1,2}


        where D is the finite differences linear operator,
        and the 2-norm is taken on the dimension of the differences.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :return: (torch.Tensor) prior :math:`g(x)`.
        """
        return torch.sum(torch.sqrt(torch.sum(self.nabla(x) ** 2, axis=-1)))

    def prox(self, x, *args, gamma=1.0, **kwargs):
        r"""Compute the proximity operator of TV with the denoiser :class:`~deepinv.models.TVDenoiser`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator at :math:`x`.
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

    :param callable negative_patch_log_likelihood: NLL function on the patch space
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

    def g(self, x, *args, **kwargs):
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


class PatchNR(nn.Module):
    r"""
    Patch prior via normalizing flows.

    The forward method evaluates its negative log likelihood.

    :param torch.nn.Module normalizing_flow: describes the normalizing flow of the model. Generally it can be any :meth:`torch.nn.Module`
        supporting backpropagation. It takes a (batched) tensor of flattened patches and the boolean rev (default `False`)
        as input and provides the value and the log-determinant of the Jacobian of the normalizing flow as an output
        If `rev=True`, it considers the inverse of the normalizing flow.
        When set to `None` it is set to a dense invertible neural network built with the FrEIA library, where the number of
        invertible blocks and the size of the subnetworks is determined by the parameters `num_layers` and `sub_net_size`.
    :param str pretrained: Define pretrained weights by its path to a `.pt` file, None for random initialization,
        `"PatchNR_lodopab_small"` for the weights from the limited-angle CT example.
    :param int patch_size: size of patches
    :param int channels: number of channels for the underlying images/patches.
    :param int num_layers: defines the number of blocks of the generated normalizing flow if `normalizing_flow` is `None`.
    :param int sub_net_size: defines the number of hidden neurons in the subnetworks of the generated normalizing flow
        if `normalizing_flow` is `None`.
    :param str device: used device
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
        super(PatchNR, self).__init__()
        if isinstance(Ff, ImportError):
            raise ImportError(
                "FrEIA is needed to use the PatchNR class. "
                "It should be installed with `pip install FrEIA`."
            ) from Ff
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

    def forward(self, x):
        r"""
        Evaluates the negative log likelihood function of th PatchNR.

        :param torch.Tensor x: image tensor
        """
        B, n_patches = x.shape[0:2]
        latent_x, logdet = self.normalizing_flow(x.view(B * n_patches, -1))
        logpz = 0.5 * torch.sum(latent_x.view(B, n_patches, -1) ** 2, -1)
        return logpz - logdet.view(B, n_patches)
