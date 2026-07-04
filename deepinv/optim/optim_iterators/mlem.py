import torch

from deepinv.utils.tensorlist import ones_like

from .optim_iterator import OptimIterator


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

        # Use one full operator for MLEM, or loop over ordered subsets for OSEM.
        if y_subsets is not None or subset_physics is not None:
            if y_subsets is None or subset_physics is None:
                raise ValueError(
                    "Both y_subsets and subset_physics must be provided together."
                )
            num_subsets = len(subset_physics)
            if num_subsets < 1:
                raise ValueError("MLEM requires at least one subset.")
            if len(y_subsets) != num_subsets:
                raise ValueError(
                    "The number of measurement subsets and physics subsets must match."
                )
            measurements = y_subsets
            physics_list = subset_physics
            prior_scale = 1.0 / num_subsets
        else:
            measurements = (y,)
            physics_list = (physics,)
            prior_scale = 1.0

        x = x_prev
        for cur_y, cur_physics in zip(measurements, physics_list, strict=True):
            # E-step/MM correction: compare measured counts to predicted counts.
            sensitivity = cur_physics.A_adjoint(ones_like(cur_y))
            if hasattr(cur_physics, "background"):
                proj = cur_physics.A(x, add_background=True)
            else:
                proj = cur_physics.A(x)

            numerator = x * cur_physics.A_adjoint(cur_y / proj.clamp(min=self.eps))
            denom = sensitivity

            # Optional OSL prior contribution in the multiplicative denominator.
            if cur_prior is not None:
                if cur_prior.__class__.__name__ == "TVPrior":
                    dx = cur_prior.nabla(x)
                    norm = torch.linalg.vector_norm(dx, ord=2, dim=-1, keepdim=True)
                    prior_grad = cur_prior.nabla_adjoint(dx / norm.clamp_min(self.eps))
                else:
                    prior_grad = cur_prior.grad(x, cur_params["g_param"])
                denom = denom + prior_scale * cur_params["lambda"] * prior_grad

            # M-step/MM minimizer: multiplicative update normalized by sensitivity.
            x = numerator / denom.clamp(min=self.eps)

        F = (
            self.cost_fn(x, cur_data_fidelity, cur_prior, cur_params, y, physics)
            if self.cost_fn is not None
            and self.has_cost
            and cur_data_fidelity is not None
            and cur_prior is not None
            else None
        )
        return {"est": (x, None), "cost": F, "it": k + 1}
