import torch
import torch.nn as nn
from deepinv.optim.optimizers import BaseOptim, create_iterator


class BaseUnfold(BaseOptim):
    r"""
    Base class for unfolded algorithms. Child of :class:`deepinv.optim.BaseOptim`.

    Enables to turn any iterative optimization algorithm into an unfolded algorithm, i.e. an algorithm
    that can be trained end-to-end, with learnable parameters. Recall that the algorithms have the
    following form (see :meth:`deepinv.optim.OptimIterator`):

    .. math::
        \begin{aligned}
        z_{k+1} &= \operatorname{step}_f(x_k, z_k, y, A, \gamma, ...)\\
        x_{k+1} &= \operatorname{step}_g(x_k, z_k, y, A, \lambda, \sigma, ...)
        \end{aligned}

    where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are learnable modules.
    These modules encompass trainable parameters of the algorithm (e.g. stepsize :math:`\gamma`, regularization parameter :math:`\lambda`, prior parameter (`g_param`) :math:`\sigma` ...)
    as well as trainable priors (e.g. a deep denoiser).

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"GD"`` (gradient descent), ``"PGD"`` (proximal gradient descent),
        ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford). See
        <optim> for more details.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
        e.g. the stepsize, regularisation parameter, denoising standard deviation.
        Each value of the dictionary can be either Iterable (distinct value for each iteration) or
        a single float (same value for each iteration).
        Default: ``{"stepsize": 1.0, "lambda": 1.0}``. See :any:`optim-params` for more details.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :meth:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: ``None``.
    :param list, deepinv.optim.Prior prior: regularization prior.
        Either a single instance (same prior for each iteration) or a list of instances of
        deepinv.optim.Prior (distinct prior for each iteration). Default: ``None``.
    :param int max_iter: number of iterations of the unfolded algorithm. Default: 5.
    :param list trainable_params: List of parameters to be trained. Each parameter should be a key of the ``params_algo``
        dictionary for the :meth:`deepinv.optim.OptimIterator` class.
        This does not encompass the trainable weights of the prior module.
    :param torch.device device: Device on which to perform the computations. Default: ``torch.device("cpu")``.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param kwargs: Keyword arguments to be passed to the :class:`deepinv.optim.BaseOptim` class.
    """

    def __init__(
        self,
        iterator,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        data_fidelity=None,
        prior=None,
        max_iter=5,
        trainable_params=["lambda", "stepsize"],
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__(
            iterator,
            max_iter=max_iter,
            data_fidelity=data_fidelity,
            prior=prior,
            params_algo=params_algo,
            **kwargs,
        )
        # Each parameter in `init_params_algo` is a list, which is converted to a `nn.ParameterList` if they should be trained.
        for param_key in trainable_params:
            if param_key in self.init_params_algo.keys():
                param_value = self.init_params_algo[param_key]
                self.init_params_algo[param_key] = nn.ParameterList(
                    [nn.Parameter(torch.tensor(el).to(device)) for el in param_value]
                )
        self.init_params_algo = nn.ParameterDict(self.init_params_algo)
        self.params_algo = self.init_params_algo.copy()
        # The prior (list of instances of :class:`deepinv.optim.Prior`) is converted to a `nn.ModuleList` to be trainable.
        self.prior = nn.ModuleList(self.prior)
        self.data_fidelity = nn.ModuleList(self.data_fidelity)


def unfolded_builder(
    iteration,
    params_algo={"lambda": 1.0, "stepsize": 1.0},
    data_fidelity=None,
    prior=None,
    max_iter=5,
    trainable_params=["lambda", "stepsize"],
    device=torch.device("cpu"),
    F_fn=None,
    g_first=False,
    **kwargs,
):
    r"""
    Helper function for building an unfolded architecture.

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"GD"`` (gradient descent), ``"PGD"`` (proximal gradient descent),
        ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford). See
        <optim> for more details.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
        e.g. the stepsize, regularisation parameter, denoising standard deviation.
        Each value of the dictionary can be either Iterable (distinct value for each iteration) or
        a single float (same value for each iteration).
        Default: ``{"stepsize": 1.0, "lambda": 1.0}``. See :any:`optim-params` for more details.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
        Either a single instance (same data-fidelity for each iteration) or a list of instances of
        :meth:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: ``None``.
    :param list, deepinv.optim.Prior prior: regularization prior.
        Either a single instance (same prior for each iteration) or a list of instances of
        deepinv.optim.Prior (distinct prior for each iteration). Default: ``None``.
    :param int max_iter: number of iterations of the unfolded algorithm. Default: 5.
    :param list trainable_params: List of parameters to be trained. Each parameter should be a key of the ``params_algo``
        dictionary for the :class:`deepinv.optim.OptimIterator` class.
        This does not encompass the trainable weights of the prior module.
    :param callable F_fn: Custom user input cost function. default: None.
    :param torch.device device: Device on which to perform the computations. Default: ``torch.device("cpu")``.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param kwargs: additional arguments to be passed to the :meth:`BaseOptim` class.
    :return: an unfolded architecture (instance of :meth:`BaseUnfold`).

    |sep|

    :Example:

    .. doctest::

        >>> import torch
        >>> import deepinv as dinv
        >>>
        >>> # Create a trainable unfolded architecture
        >>> model = dinv.unfolded.unfolded_builder(
        ...     iteration="PGD",
        ...     data_fidelity=dinv.optim.L2(),
        ...     prior=dinv.optim.PnP(dinv.models.DnCNN(in_channels=1, out_channels=1, train=True)),
        ...     params_algo={"stepsize": 1.0, "g_param": 1.0},
        ...     trainable_params=["stepsize", "g_param"]
        ... )
        >>> # Forward pass
        >>> x = torch.randn(1, 1, 16, 16)
        >>> physics = dinv.physics.Denoising()
        >>> y = physics(x)
        >>> x_hat = model(y, physics)


    """
    iterator = create_iterator(iteration, prior=prior, F_fn=F_fn, g_first=g_first)
    return BaseUnfold(
        iterator,
        max_iter=max_iter,
        trainable_params=trainable_params,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        prior=prior,
        params_algo=params_algo,
        device=device,
        **kwargs,
    )
