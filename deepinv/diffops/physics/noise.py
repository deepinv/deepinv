import torch
from deepinv.diffops.physics.forward import DecomposablePhysics


class Denoising(DecomposablePhysics):
    def __init__(self, sigma=.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_model = GaussianNoise(sigma)


class GaussianNoise(torch.nn.Module):
    r'''
    Gaussian noise module. It can be added to a physics operator by setting
    physics.noise = GaussianNoise()
    '''
    def __init__(self, sigma=.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return x + torch.randn_like(x)*self.sigma


class PoissonNoise(torch.nn.Module):
    r'''
    Poisson noise module. It can be added to a physics operator by setting
    physics.noise = PoissonNoise()
    '''
    def __init__(self, gain=1., normalize=True):
        '''
        Poisson noise is defined as
        :math:`y = \mathcal{P}(\gamma x)`

        where :math:`\gamma` is the gain. If normalize=True, the output :math:`y` is postmultiplied by the gain.
        '''
        super().__init__()
        self.gain = gain
        self.normalize = normalize

    def forward(self, x):
        y = torch.poisson(self.gain*x)
        if self.normalize:
            y /= self.gain
        return y


class PoissonGaussianNoise(torch.nn.Module):
    r'''
    Poisson-Gaussian noise module. It can be added to a physics operator by setting
    physics.noise = PoissonGaussianNoise()
    '''
    def __init__(self, gain=1., sigma=.1):
        '''
        Poisson noise is defined as
        :math:`y = \frac{z}{\gamma} + \epsilon`

        where :math:`z\sim\mathcal{P}(\gamma x)` and :math:`\epsilon\sim\mathcal{N}(0, I \sigma^2)`.
        '''
        super().__init__()
        self.gain = gain
        self.sigma = sigma

    def forward(self, x):
        y = torch.poisson(self.gain*x)/self.gain

        y += torch.randn_like(x)*self.sigma
        return y


class UniformNoise(torch.nn.Module):
    r'''
    Uniform (additive) noise module. It can be added to a physics operator by setting
    physics.noise = UniformNoise()
    '''
    def __init__(self, a=.1):
        '''
        Uniform noise is defined as
        :math:`y = x + \epsilon`

        where :math:`\epsilon\sim\mathcal{U}(-a,a)`.
        '''
        super().__init__()
        self.amplitude = a

    def forward(self, x):
        return x + (self.rand_like(x)-.5)*2*self.amplitude

