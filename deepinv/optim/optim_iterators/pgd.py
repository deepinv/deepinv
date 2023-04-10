from .optim_iterator import OptimIterator, fStep, gStep

class PGDIteration(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
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
            return x - cur_params['stepsize'] * cur_params['lambda'] * self.data_fidelity.grad(x, y, physics)
        else:
            return self.data_fidelity.prox(x, y, physics, 1/(cur_params['lambda'] * cur_params['stepsize']))


class gStepPGD(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepPGD, self).__init__(**kwargs)

    def forward(self, x, cur_params):
        if not self.g_first:
            return self.prox_g(x, cur_params['g_param'])
        else:
            return x - cur_params['stepsize'] * self.grad_g(x, cur_params['g_param'])


