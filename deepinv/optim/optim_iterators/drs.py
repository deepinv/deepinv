from .optim_iterator import OptimIterator, fStep, gStep

class DRSIteration(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(DRSIteration, self).__init__(**kwargs)
        self.g_step = gStepDRS(**kwargs)
        self.f_step = fStepDRS(**kwargs)

    def forward(self, X, cur_params, y, physics):
        '''
        Adapts the generic forward class to the DRS case as an additional relaxation (averaging) step is required before
        the usual relaxation step.
        '''
        x_prev = X['est'][0]
        if not self.g_first:
            x = self.f_step(x_prev, y, physics, cur_params)
            x = self.g_step(x, cur_params)
        else:
            x = self.g_step(x_prev, cur_params)
            x = self.f_step(x, y, physics, cur_params)
        x = (x_prev + x) / 2.
        x = self.relaxation_step(x, x_prev)
        F = self.F_fn(x,cur_params,y,physics) if self.F_fn else None
        return {'est': (x, ), 'cost': F}


class fStepDRS(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepDRS, self).__init__(**kwargs)

    def forward(self, x, y, physics, cur_params):
        return 2 * self.data_fidelity.prox(x, y, physics, 1 / (self.lamb * cur_params['stepsize'])) - x


class gStepDRS(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepDRS, self).__init__(**kwargs)

    def forward(self, z, cur_params):
        return 2 * self.prox_g(z, cur_params['g_param']) - z


