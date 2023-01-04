import torch


class GaussianNoise(torch.nn.Module): # parent class for forward models
    def __init__(self, std=.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        return x + torch.randn_like(x)*self.std


class Forward(torch.nn.Module):  # parent class for forward models
    def __init__(self, A=lambda x: x, A_adjoint=lambda x: x,
                 noise_model=lambda x: x, sensor_model=lambda x: x):
        super().__init__()
        self.noise_model = noise_model
        self.sensor_model = sensor_model
        self.forw = A
        self.adjoint = A_adjoint

    def __add__(self, other): #  physics3 = physics1 + physics2
        A = lambda x: self.A(other.A(x)) # (A' = A_1 A_2)
        A_adjoint = lambda x: other.A_adjoint(self.A_adjoint(x)) #(A'^{T} = A_2^{T} A_1^{T})
        noise = self.noise_model
        sensor = self.sensor_model
        return Forward(A, A_adjoint, noise, sensor)

    def forward(self, x):# degrades signal
        return self.sensor(self.noise(self.A(x)))

    def A(self, x):
        return self.forw(x)

    def sensor(self, x):
        return self.sensor_model(x)

    def noise(self, x):
        return self.noise_model(x)

    def A_adjoint(self, x):
        return self.adjoint(x)

    def A_dagger(self, x): # degrades signal
        # USE Conjugate gradient here as default option min_x |A(x)-y| (we need A and A^{T})
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