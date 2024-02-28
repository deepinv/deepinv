import torch
import torch.nn as nn

from deepinv.optim.utils import gradient_descent
from deepinv.models.tv import TVDenoiser


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

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float gamma: stepsize of the proximity operator.
        :param float lamb: math:`\lambda` parameter in front of :math:`f`
        :return: (torch.tensor) proximity operator :math:`\operatorname{prox}_{\gamma \lambda g)^*}(x)`, computed in :math:`x`.
        """
        return x - gamma * self.prox(x / gamma, lamb / gamma, *args, **kwargs)


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

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
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

    def forward(self, x, sigma):
        r"""
        Applies the denoiser to the input signal.

        :param torch.Tensor x: the input tensor.
        :param float sigma: the noise level.
        """
        return (1 / sigma**2) * (x - self.denoiser(x, sigma))


class Tikhonov(Prior):
    r"""
    Tikhonov regularizer :math:`\reg{x} = \frac{1}{2}\| x \|_2^2`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, ths=1.0):
        r"""
        Computes the Tikhonov regularizer :math:`\reg{x} = \frac{\tau}{2}\| x \|_2^2`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :param float ths: regularization parameter :math:`\tau`.
        :return: (torch.Tensor) prior :math:`\reg{x}`.
        """
        return (
            0.5
            * ths
            * torch.norm(x.contiguous().view(x.shape[0], -1), p=2, dim=-1) ** 2
        )

    def grad(self, x):
        r"""
        Calculates the gradient of the Tikhonov regularization term :math:`\regname` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :return: (torch.Tensor) gradient at :math:`x`.
        """
        return x

    def prox(self, x, ths=1.0, gamma=1.0):
        r"""
        Calculates the proximity operator of the Tikhonov regularization term :math:`\gamma \tau g` at :math:`x`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float ths: regularization parameter :math:`\tau`.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """
        return (1 / (ths * gamma + 1)) * x


class L1Prior(Prior):
    r"""
    :math:`\ell_1` prior :math:`\reg{x} = \| x \|_1`.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def g(self, x, ths=1.0):
        r"""
        Computes the regularizer :math:`\reg{x} = \tau\| x \|_1`.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :param float ths: threshold parameter :math:`\tau`.
        :return: (torch.Tensor) prior :math:`\reg{x}`.
        """
        return ths * torch.norm(x.contiguous().view(x.shape[0], -1), p=1, dim=-1)

    def prox(self, x, ths=1.0, gamma=1.0):
        r"""
        Calculates the proximity operator of the l1 regularization term :math:`\regname` at :math:`x`.

        More precisely, it computes

        .. math::
            \operatorname{prox}_{\gamma \tau g}(x) = \operatorname{sign}(x) \max(|x| - \gamma \tau, 0)


        where :math:`\tau` is the threshold parameter and :math:`\gamma` is a stepsize.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float ths: threshold parameter :math:`\tau`.
        :param float gamma: stepsize of the proximity operator.
        :return torch.Tensor: proximity operator at :math:`x`.
        """
        return torch.sign(x) * torch.max(
            torch.abs(x) - ths * gamma, torch.zeros_like(x)
        )


class TVPrior(Prior):
    r"""
    Total variation (TV) prior :math:`\reg{x} = \| D x \|_{1,2}`.
    """

    def __init__(self, def_crit=1e-8, n_it_max=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True
        self.TVModel = TVDenoiser(crit=def_crit, n_it_max=n_it_max)

    def g(self, x, ths=1.0, **kwargs):
        r"""
        Computes the regularizer

        .. math::
             \tau g(x) = \tau \|Dx\|_{1,2}


        where D is the finite differences linear operator,
        and the 2-norm is taken on the dimension of the differences.

        :param torch.Tensor x: Variable :math:`x` at which the prior is computed.
        :param torch.Tensor, float ths: Regularization parameter :math:`\tau` in the proximal operator (default value = 1.0).
        :return: (torch.Tensor) prior :math:`\tau g(x)`.
        """
        return ths * torch.sum(torch.sqrt(torch.sum(self.nabla(x) ** 2, axis=-1)))

    def prox(self, x, ths=1.0, gamma=1.0, *args, **kwargs):
        r"""Compute the proximity operator of TV with the denoiser :class:`~deepinv.models.TVDenoiser`.

        :param torch.Tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param float ths: threshold parameter :math:`\tau`.
        :param float gamma: stepsize of the proximity operator.
        :return: (torch.Tensor) proximity operator at :math:`x`.
        """
        return self.TVModel(x, ths=ths * gamma)

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
