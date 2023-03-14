from .optim_iterator import OptimIterator, fStep, gStep

class DRS(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(DRS, self).__init__(**kwargs)
        self.g_step = gStepDRS(**kwargs)
        self.f_step = fStepDRS(**kwargs)


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


