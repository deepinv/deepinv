import torch
import torch.nn as nn
from deepinv.optim.optim_iterators import *
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import BaseOptim, str_to_class


class BaseUnfold(BaseOptim):
    r"""
    Base class for unfolded algorithms. Child of :class:`deepinv.optim.BaseOptim`.

    Enables to turn any iterative optimization algorithm into an unfolded algorithm, i.e. an algorithm
    that can be trained end-to-end, with learnable parameters. Recall that the algorithms have the
    following form (see :meth:`deepinv.optim.optim_iterators.BaseIterator`):

    .. math::
        \begin{aligned}
        z_{k+1} &= \operatorname{step}_f(x_k, z_k, y, A, \lambda, \gamma, ...)\\
        x_{k+1} &= \operatorname{step}_g(x_k, z_k, y, A, \sigma, ...)
        \end{aligned}

    where :math:`\operatorname{step}_f` and :math:`\operatorname{step}_g` are learnable modules. 
    These modules encompass trainable parameters of the algorithm (e.g. stepsize :math:`\gamma`, regularization parameter :math:`\lambda`, prior parameter (`g_param`) :math:`\sigma` ...)
    as well as trainable priors (e.g. a deep denoiser).

    :param list trainable_params: List of parameters to be trained. Each parameter should be a key of the `params_algo` dictionary for the :class:`deepinv.optim.optim_iterators.BaseIterator` class.
                    This does not encompass the trainable weights of the prior module . 
    :param torch.device device: Device on which to perform the computations. Default: `torch.device("cpu")`.
    :param args:  Non-keyword arguments to be passed to the :class:`deepinv.optim.BaseOptim` class.
    :param kwargs: Keyword arguments to be passed to the :class:`deepinv.optim.BaseOptim` class.
    """

    def __init__(self, *args, trainable_params=[], device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        # Each parameter in `init_params_algo` is a list, which is converted to a `nn.ParameterList` if they should be trained.
        for param_key in trainable_params:
            if param_key in self.init_params_algo.keys():
                param_value = self.init_params_algo[param_key]
                self.init_params_algo[param_key] = nn.ParameterList(
                    [nn.Parameter(torch.tensor(el).to(device)) for el in param_value]
                )
        self.init_params_algo = nn.ParameterDict(self.init_params_algo)
        self.params_algo = self.init_params_algo.copy()
        # The prior (list of instances of :class:`deepinv.optim.Prior) is converted to a `nn.ModuleList` to be trainable.
        self.prior = nn.ModuleList(self.prior)


def unfolded_builder(
    iteration, data_fidelity=L2(), F_fn=None, g_first=False, beta=1.0, **kwargs
):
    r"""
    Function building the appropriate Unfolded architecture. 

    :param str, deepinv.optim.optim_iterators.OptimIterator iteration: either the name of the algorithm to be used, or an optim iterator .
        If an algorithm name (string), should be either `"PGD"`, `"ADMM"`, `"HQS"`, `"CP"` or `"DRS"`.
    :param deepinv.optim.DataFidelity data_fidelity: data fidelity term in the optimization problem.
    :param callable F_fn: Custom user input cost function. default: None.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. default: False
    :param float beta: relaxation parameter in the fixed point algorithm. Default: `1.0`.
    """
    # If no custom objective function F_fn is given but g is explicitly given, we have an explicit objective function.
    explicit_prior = (
        kwargs["prior"][0].explicit_prior
        if isinstance(kwargs["prior"], list)
        else kwargs["prior"].explicit_prior
    )
    if F_fn is None and explicit_prior:

        def F_fn(x, prior, cur_params, y, physics):
            return cur_params["lambda"] * data_fidelity(x, y, physics) + prior.g(
                x, cur_params["g_param"]
            )

        has_cost = (
            True
        )  # boolean to indicate if there is a cost function to evaluate along the iterations
    else:
        has_cost = False

    # Create a instance of :class:`deepinv.optim.optim_iterators.OptimIterator`.
    # If the iteration is directly given as an instance of OptimIterator, nothing to do
    if isinstance(
        iteration, str
    ):  # If the name of the algorithm is given as a string, the correspondong class is automatically called.
        iterator_fn = str_to_class(iteration + "Iteration")
        iteration = iterator_fn(
            data_fidelity=data_fidelity,
            g_first=g_first,
            beta=beta,
            F_fn=F_fn,
            has_cost=has_cost,
        )
    return BaseUnfold(iteration, has_cost=has_cost, **kwargs)
