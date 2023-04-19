from .optim_iterator import OptimIterator, fStep, gStep

class HQSIteration(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(HQSIteration, self).__init__(**kwargs)
        self.g_step = gStepHQS(**kwargs)
        self.f_step = fStepHQS(**kwargs)
        self.requires_prox_g = True

class fStepHQS(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepHQS, self).__init__(**kwargs)

    def forward(self, x, cur_params, y, physics):
        return self.data_fidelity.prox(x, y, physics, 1/(cur_params['lambda'] * cur_params['stepsize']))


class gStepHQS(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepHQS, self).__init__(**kwargs)

    def forward(self, x, prior, cur_params):
        return prior['prox_g'](x, cur_params['g_param'])


