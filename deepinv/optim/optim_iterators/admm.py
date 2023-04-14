import torch
from .optim_iterator import OptimIterator, fStep, gStep

class ADMMIteration(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(ADMMIteration, self).__init__(**kwargs)
        self.g_step = gStepADMM(**kwargs)
        self.f_step = fStepADMM(**kwargs)

    def forward(self, X, cur_prior, cur_params, y, physics):

        x, z = X['est']

        if z.shape != x.shape:  # In ADMM, the "dual" variable u is a fake dual variable as it lives in the primal, hence this line to prevent from usual initialisation
            z = torch.zeros_like(x)

        z_prev = z.clone()

        z_temp = self.g_step(x, z, cur_prior, cur_params)
        x = self.f_step(z_temp, z, y, physics, cur_params)
        z = z_prev + self.beta*(z_temp - x)

        F = self.F_fn(x, cur_params, y, physics) if self.F_fn else None
        return {'est': (x,z), 'cost': F}

class fStepADMM(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepADMM, self).__init__(**kwargs)

    def forward(self, x, u, y, physics, cur_params):
        return self.data_fidelity.prox(x+u, y, physics, 1/(cur_params['lambda']*cur_params['stepsize']))


class gStepADMM(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepADMM, self).__init__(**kwargs)

    def forward(self, x, z, cur_prior, cur_params):
        return cur_prior['prox_g'](x-z, cur_params['g_param'])
