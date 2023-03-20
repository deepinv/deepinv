from .optim_iterator import OptimIterator, fStep, gStep

class HQSIteration(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(HQSIteration, self).__init__(**kwargs)
        self.g_step = gStepHQS(**kwargs)
        self.f_step = fStepHQS(**kwargs)


class fStepHQS(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepHQS, self).__init__(**kwargs)

    def forward(self, x, y, physics, it):
        return self.data_fidelity.prox(x, y, physics, self.lamb * self.stepsize[it])


class gStepHQS(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepHQS, self).__init__(**kwargs)

    def forward(self, x, it):
        return self.prox_g(x, self.g_param[it], it)


