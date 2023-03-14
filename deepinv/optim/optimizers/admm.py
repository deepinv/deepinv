import torch
from .optim_iterator import OptimIterator, fStep, gStep

class ADMM(OptimIterator):

    def __init__(self, **kwargs):
        '''
        TODO: add doc
        '''
        super(ADMM, self).__init__(**kwargs)
        self.g_step = gStepADMM(**kwargs)
        self.f_step = fStepADMM(**kwargs)

    def forward(self, z_u, it, y, physics):

        z, u = z_u

        if u.shape != z.shape:  # In ADMM, the "dual" variable u is a fake dual variable as it lives in the primal, hence this line to prevent from usual initialisation
            u = torch.zeros_like(z)

        u_prev = u.clone()

        x = self.g_step(z, u, it)
        z = self.f_step(x, u, y, physics, it)
        u = u_prev + self.beta*(x - z)

        z_u = (z, u)

        return z_u

class fStepADMM(fStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(fStepADMM, self).__init__(**kwargs)

    def forward(self, x, u, y, physics, it):
        return self.data_fidelity.prox(x+u, y, physics, self.lamb*self.stepsize[it])


class gStepADMM(gStep):

    def __init__(self, **kwargs):
        """
        TODO: add doc
        """
        super(gStepADMM, self).__init__(**kwargs)

    def forward(self, z, u, it):
        return self.prox_g(z-u, self.g_param[it], it)
