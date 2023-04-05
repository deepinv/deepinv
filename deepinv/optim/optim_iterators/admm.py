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

    def forward(self, z_u, cur_params, y, physics):

        z, u = z_u

        if u.shape != z.shape:  # In ADMM, the "dual" variable u is a fake dual variable as it lives in the primal, hence this line to prevent from usual initialisation
            u = torch.zeros_like(z)

        u_prev = u.clone()

        x = self.g_step(z, u, cur_params)
        z = self.f_step(x, u, cur_params, y, physics)
        u = u_prev + self.beta*(x - z)

        z_u = (z, u)

        return z_u

class fStepADMM(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepADMM, self).__init__(**kwargs)

    def forward(self, x, cur_params, u, y, physics):
        return self.data_fidelity.prox(x+u, y, physics, 1/(self.lamb*cur_params['stepsize']))


class gStepADMM(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepADMM, self).__init__(**kwargs)

    def forward(self, z, cur_params, u):
        return self.prox_g(z-u, cur_params['g_param'])
