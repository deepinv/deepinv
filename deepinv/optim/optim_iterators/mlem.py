from .optim_iterator import OptimIterator
from torch import ones_like


class MLEMIteration(OptimIterator):
    """
    Iterator for the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm for Poisson inverse problems.

    Class for a single iteration of the MLEM algorithm :footcite:t:`sheppMaximumLikelihoodReconstruction1982`,
    which is a classic baseline reconstruction method for inverse problems with Poisson noise statistics.
    More details on the algorithm can be found in the documentation of the :class:`deepinv.optim.optimizers.MLEM` optimizer.
    """

    def __init__(self, **kwargs):
        super(MLEMIteration, self).__init__(**kwargs)

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single Maximum-Likelihood Expectation-Maximization (MLEM) iteration.

        This corresponds to an update on both the Poisson negative log-likelihood and prior terms if a prior is provided, and only on Poisson negative log-likelihood otherwise.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        """
        x_prev = X["est"][0]
        k = 0 if "it" not in X else X["it"]
        sensitivity = physics.A_adjoint(ones_like(y))

        x = x_prev * physics.A_adjoint(y / (physics.A(x_prev).clamp(min=1e-15)))

        if cur_prior is not None:
            denom = sensitivity + cur_params["lambda"] * cur_prior.grad(
                x, cur_params["g_param"]
            )
        else:
            denom = sensitivity

        x = x / denom.clamp(min=1e-15)
        F = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.cost_fn is not None
            and self.has_cost
            and cur_data_fidelity is not None
            and cur_prior is not None
            else None
        )
        return {"est": (x, None), "cost": F, "it": k + 1}
