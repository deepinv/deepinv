from .optim_iterator import OptimIterator, fStep, gStep

class PD(OptimIterator):

    def __init__(self, **kwargs):
        '''
        In this case the algorithm works on the product space HxH^* so input/output variable is a concatenation of
        primal and dual variables.

        TODO:
        - check that there is no conflict with the data_fidelity.prox
        - check that there is freedom in how to apply replacement of prox operators (see J. Adler works)
        '''
        super(PD, self).__init__(**kwargs)
        self.g_step = gStepPD(**kwargs)
        self.f_step = fStepPD(**kwargs)

    def forward(self, pd_var, it, y, physics):

        x, u = pd_var

        x_ = self.g_step(x, physics.A_adjoint(u), it)
        Ax_cur = physics.A(2 * x_ - x)
        u_ = self.f_step(Ax_cur, u, y, it)

        pd_variable = (x_, u_)

        return pd_variable



class fStepPD(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepPD, self).__init__(**kwargs)

    def forward(self, Ax_cur, u, y, it):  # Beware this is not the prox of f(A\cdot) but only the prox of f, A is tackled independently in PD
       v = u + self.stepsize[it] * Ax_cur
       return v - self.stepsize[it] * self.data_fidelity.prox_norm(v / self.stepsize[it], y, self.lamb)


class gStepPD(gStep):

    def __init__(self, stepsize_2=[1.] * 50, **kwargs):
        """
        TODO: add doc
        """
        super(gStepPD, self).__init__(**kwargs)
        self.stepsize_2 = stepsize_2

    def forward(self, x, Atu, it):
        return self.prox_g(x - self.stepsize_2[it] * Atu, self.stepsize_2[it] * self.g_param[it], it)
