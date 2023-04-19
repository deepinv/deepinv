from .optim_iterator import OptimIterator, fStep, gStep
from .utils import gradient_descent_step


class PGDIteration(OptimIterator):
    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(PGDIteration, self).__init__(**kwargs)
        self.g_step = gStepPGD(**kwargs)
        self.f_step = fStepPGD(**kwargs)


class fStepPGD(fStep):
    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepPGD, self).__init__(**kwargs)

    def forward(self, x, cur_params, y, physics):
        if not self.g_first:
            grad = cur_params["stepsize"] * self.data_fidelity.grad(x, y, physics)
            return gradient_descent_step(x, grad, self.bregman_potential)
        else:
            return self.data_fidelity.prox(
                x, y, physics, 1 / (cur_params["lambda"] * cur_params["stepsize"])
            )


class gStepPGD(gStep):
    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepPGD, self).__init__(**kwargs)

    def forward(self, x, prior, cur_params):
        if not self.g_first:
            return prior["prox_g"](x, cur_params["g_param"])
        else:
            grad = cur_params["stepsize"] * prior["grad_g"](x, cur_params["g_param"])
            return gradient_descent_step(x, grad, self.bregman_potential)
