from .optim_iterator import OptimIterator, fStep, gStep
import torch


class MLEMIteration(OptimIterator):
    def __init__(self, **kwargs):
        super(MLEMIteration, self).__init__(**kwargs)
        self.g_step = gStepMLEM(**kwargs)
        self.f_step = fStepMLEM(**kwargs)

    def forward(
        self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        r"""
        Single Maximum-Likelihood Expectation-Maximization (MLEM) iteration.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        """
        x_prev = X["est"][0]
        k = 0 if "it" not in X else X["it"]
        sensitivity = physics.A_adjoint(torch.ones_like(y))
        x = self.f_step(x_prev, cur_data_fidelity, cur_params, y, physics)
        if cur_prior is not None:
            denom = sensitivity + self.g_step(x, cur_prior, cur_params)
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


class fStepMLEM(fStep):
    def __init__(self, **kwargs):
        super(fStepMLEM, self).__init__(**kwargs)

    def forward(self, x, cur_data_fidelity, cur_params, y, physics):
        return x * physics.A_adjoint(y / (physics.A(x).clamp(min=1e-15)))


class gStepMLEM(gStep):
    def __init__(self, **kwargs):
        super(gStepMLEM, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda \regname`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        prox_g = cur_prior.prox(
            x,
            cur_params["g_param"],
            gamma=cur_params["lambda"] * cur_params["stepsize"],
        )
        return x - prox_g
