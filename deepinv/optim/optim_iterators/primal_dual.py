from .optim_iterator import OptimIterator, fStep, gStep

class PDIteration(OptimIterator):

    def __init__(self, **kwargs):
        r'''
        TODO
        '''
        super(PDIteration, self).__init__(**kwargs)
        self.g_step = gStepPD(**kwargs)
        self.f_step = fStepPD(**kwargs)

    def forward(self, X, cur_params, y, physics):

        x_prev, u_prev = X['est']
        F_prev = X['cost']

        x = self.g_step(x_prev, physics.A_adjoint(u_prev), cur_params)
        u = self.f_step(physics.A(2 * x - x_prev), u_prev, cur_params, y)
        F = self.F_fn(x,cur_params, y, physics) if self.F_fn else None

        return {'est': (x,u), 'cost': F}



class fStepPD(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepPD, self).__init__(**kwargs)

    def forward(self, Ax_cur, u, y, it):  # Beware this is not the prox of f(A\cdot) but only the prox of f, A is tackled independently in PD
       v = u + self.stepsize[it] * Ax_cur
       return v - self.stepsize[it] * self.data_fidelity.prox_f(v, y, 1 / (self.stepsize[it]*self.lamb))


class gStepPD(gStep):

    def __init__(self, stepsize=[1.] * 50, **kwargs):
        """
        TODO: add doc
        """
        super(gStepPD, self).__init__(**kwargs)
        self.stepsize = stepsize

    def forward(self, x, Atu, it):
        return self.prox_g(x - self.stepsize[it] * Atu, self.stepsize[it] * self.g_param[it], it)
