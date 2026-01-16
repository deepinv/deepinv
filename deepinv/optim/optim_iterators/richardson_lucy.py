from .optim_iterator import OptimIterator, fStep, gStep
import torch


class RichardsonLucyIteration(OptimIterator):
    def __init__(self, **kwargs):
        super(RichardsonLucyIteration, self).__init__(**kwargs)
        self.g_step = gStepRichardsonLucy(**kwargs)
        self.f_step = fStepRichardsonLucy(**kwargs)
        self.sensitivity = kwargs.get("physics", None)(
            torch.ones_like(kwargs.get("y", torch.tensor(0.0)))
        )

    def forward(self, x, cur_data_fidelity, cur_prior, cur_params, y, physics):
        r"""
        Single Richardson-Lucy iteration.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param deepinv.optim.DataFidelity cur_data_fidelity: Instance of the DataFidelity class defining the current data_fidelity.
        :param deepinv.optim.Prior cur_prior: Instance of the Prior class defining the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: Instance of the physics modeling the data-fidelity term.
        """

        x = self.f_step(x, cur_data_fidelity, cur_params, y, physics)
        if cur_prior is not None:
            denom = self.sensitivity + self.g_step(x, cur_prior, cur_params)
        else:
            denom = self.sensitivity

        return x / denom.clamp(min=1e-15)


class fStepRichardsonLucy(fStep):
    def __init__(self, **kwargs):
        super(fStepRichardsonLucy, self).__init__(**kwargs)

    def forward(self, x, y, physics):
        return x * physics.A_adjoint(y / (physics.A(x).clamp(min=1e-15)))


class gStepRichardsonLucy(gStep):
    def __init__(self, **kwargs):
        super(gStepRichardsonLucy, self).__init__(**kwargs)

    def forward(self, x, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`\lambda \regname`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current parameters of the algorithm.
        """
        prox_g = cur_prior.prox(
            x,
            cur_params["g_param"],
            gamma=cur_params["lambda"],
        )

        return x - prox_g
