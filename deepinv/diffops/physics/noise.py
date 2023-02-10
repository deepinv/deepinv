import torch

class GaussianNoise(torch.nn.Module): # parent class for forward models
    def __init__(self, sigma=.1):
        super().__init__()
        self.std = sigma

    def forward(self, x):
        return x + torch.randn_like(x)*self.std