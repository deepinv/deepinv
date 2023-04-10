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

        x = self.g_step(x_prev, physics.A_adjoint(u_prev), cur_params)
        u = self.f_step(physics.A(2 * x - x_prev), u_prev, y, cur_params)

        F = self.F_fn(x, cur_params, y, physics) if self.F_fn else None

        return {'est': (x, u), 'cost': F}



class fStepPD(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepPD, self).__init__(**kwargs)

    def forward(self, Ax_cur, u, y, cur_params):  # Beware this is not the prox of f(A\cdot) but only the prox of f, A is tackled independently in PD
       v = u + cur_params['stepsize'] * Ax_cur
       return v - cur_params['stepsize'] * self.data_fidelity.prox_f(v, y, 1 / (cur_params['stepsize']*cur_params['lambda']))


class gStepPD(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepPD, self).__init__(**kwargs)

    def forward(self, x, Atu, cur_params):
        return self.prox_g(x - cur_params['stepsize'] * Atu, cur_params['g_param'])  # Beware, this is not correct: we should have a product of stepsize / reg param
