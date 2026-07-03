from .optim_iterator import OptimIterator
from deepinv.utils.tensorlist import TensorList, ones_like


def _clamp_min(x, min_value):
    if isinstance(x, TensorList):
        return TensorList([xi.clamp(min=min_value) for xi in x])
    return x.clamp(min=min_value)


class MLEMIteration(OptimIterator):
    r"""
    Iterator for the Maximum-Likelihood Expectation-Maximization (MLEM) algorithm for Poisson inverse problems.

    Class for a single iteration of the MLEM algorithm :footcite:t:`sheppMaximumLikelihoodReconstruction1982`,
    which is a classic baseline reconstruction method for inverse problems with Poisson noise statistics.
    More details on the algorithm can be found in the documentation of the :class:`deepinv.optim.optimizers.MLEM` optimizer.
    """

    def __init__(self, eps: float = 1e-15, **kwargs):
        self.eps = eps
        super(MLEMIteration, self).__init__(**kwargs)

    def _mlem_update(
        self,
        x_prev,
        cur_prior,
        cur_params,
        y,
        physics,
        prior_scale: float = 1.0,
    ):
        sensitivity = physics.A_adjoint(ones_like(y))
        if hasattr(physics, "background"):
            proj = physics.A(x_prev, add_background=True)
        else:
            proj = physics.A(x_prev)
        x = x_prev * physics.A_adjoint(y / _clamp_min(proj, self.eps))

        if cur_prior is not None:
            denom = sensitivity + prior_scale * cur_params["lambda"] * cur_prior.grad(
                x, cur_params["g_param"]
            )
        else:
            denom = sensitivity

        return x / denom.clamp(min=self.eps)

    def _ordered_subsets_update(
        self, x_prev, cur_prior, cur_params, y_subsets, subset_physics
    ):
        num_subsets = len(subset_physics)
        if num_subsets < 1:
            raise ValueError("MLEM requires at least one subset.")

        num_measurements = len(y_subsets)
        if num_measurements != num_subsets:
            raise ValueError(
                "The number of measurement subsets and physics subsets must match."
            )

        x = x_prev
        prior_scale = 1.0 / num_subsets
        for i in range(num_subsets):
            x = self._mlem_update(
                x,
                cur_prior,
                cur_params,
                y_subsets[i],
                subset_physics[i],
                prior_scale=prior_scale,
            )
        return x

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

        y_subsets = kwargs.get("y_subsets")
        subset_physics = kwargs.get("subset_physics")
        if y_subsets is not None or subset_physics is not None:
            if y_subsets is None or subset_physics is None:
                raise ValueError(
                    "Both y_subsets and subset_physics must be provided together."
                )
            x = self._ordered_subsets_update(
                x_prev, cur_prior, cur_params, y_subsets, subset_physics
            )
        else:
            x = self._mlem_update(x_prev, cur_prior, cur_params, y, physics)

        F = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.cost_fn is not None
            and self.has_cost
            and cur_data_fidelity is not None
            and cur_prior is not None
            else None
        )
        return {"est": (x, None), "cost": F, "it": k + 1}
