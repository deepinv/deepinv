from .optim_iterator import OptimIterator, fStep, gStep

class DRSIteration(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(DRSIteration, self).__init__(**kwargs)
        self.g_step = gStepDRS(**kwargs)
        self.f_step = fStepDRS(**kwargs)

    def forward(self, x, it, y, physics):
        '''
        Adapts the generic forward class to the DRS case as an additional relaxation (averaging) step is required before
        the usual relaxation step.
        '''
        x_prev = x[0]
        if not self.g_first:
            x = self.f_step(x_prev, y, physics, it)
            x = self.g_step(x, it)
        else:
            x = self.g_step(x_prev, it)
            x = self.f_step(x, y, physics, it)
        x = (x_prev + x) / 2.
        x = self.relaxation_step(x, x_prev)
        return (x,)


class fStepDRS(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepDRS, self).__init__(**kwargs)

    def forward(self, x, y, physics, it):
        return 2 * self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it]) - x


class gStepDRS(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepDRS, self).__init__(**kwargs)

    def forward(self, z, it):
        return 2 * self.prox_g(z, self.g_param[it], it) - z


