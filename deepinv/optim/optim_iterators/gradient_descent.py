from .optim_iterator import OptimIterator, fStep, gStep
from .utils import gradient_descent_step


class GDIteration(OptimIterator):
    def __init__(self, **kwargs):
        r"""
        TODO
        """
        super(GDIteration, self).__init__(**kwargs)
        self.g_step = gStepGD(**kwargs)
        self.f_step = fStepGD(**kwargs)
        self.requires_grad_g = True

    def forward(self, X, cur_prior, cur_params, y, physics):
        """ """
        x_prev = X["est"][0]
        grad = cur_params["stepsize"] * (
            self.g_step(x_prev, cur_prior, cur_params)
            + self.f_step(x_prev, cur_params, y, physics)
        )
        x = gradient_descent_step(x_prev, grad, self.bregman_potential)
        x = self.relaxation_step(x, x_prev)
        F = self.F_fn(x, cur_params, y, physics) if self.F_fn else None
        return {"est": (x,), "cost": F}


class fStepGD(fStep):
    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepGD, self).__init__(**kwargs)

    def forward(self, x, cur_params, y, physics):
        return cur_params["lambda"] * self.data_fidelity.grad(x, y, physics)


class gStepGD(gStep):
    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepGD, self).__init__(**kwargs)

    def forward(self, x, prior, cur_params):
        return prior["grad_g"](x, cur_params["g_param"])
