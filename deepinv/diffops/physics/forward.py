import torch

class GaussianNoise(torch.nn.Module): # parent class for forward models
    def __init__(self, std=.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x)*self.std

class Forward(torch.nn.Module): # parent class for forward models
    def __init__(self, A = lambda x: x, A_adjoint = lambda x: x, noise_model = GaussianNoise(std=0.2)):
        super().__init__()
        self.noise_model = noise_model
        self.forw = A
        self.adjoint = A_adjoint

    def __add__(self, other):
        A = lambda x: self.A(other.A(x))
        A_adjoint = lambda x: other.A_adjoint(self.A_adjoint(x))
        noise = self.noise_model
        return Forward(A, A_adjoint, noise)

    def forward(self, x): # degrades signal
        return self.noise(self.A(x))

    def A(self, x):
        return self.forw(x)

    def noise(self, x):
        return self.noise_model(x)

    def A_adjoint(self, x):
        return self.adjoint(x)

    def A_dagger(self, x): # degrades signal
        # USE Conjugate gradient here as default option
        return self.A_adjoint(x)


class Denoising(Forward):
    def __init__(self):
        super().__init__()
        self.name = 'denoising'

    def A(self, x):
        return x

    def A_dagger(self, x):
        return x

    def A_adjoint(self, x):
        return x