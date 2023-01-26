import torch
import torch.nn as nn

class ProximalGradient(nn.Module):
    """
    TODO: add doc
    """

    def __init__(self, denoiser, denoise_level=None, max_iter=10, gamma=1, crit_conv=1e-4):
        super(ProximalGradient, self).__init__()

        self.denoiser = denoiser
        self.denoise_level = denoise_level
        self.max_iter = max_iter
        self.gamma = gamma
        self.crit_conv = crit_conv

    def forward(self, y, physics):
        """
        TODO: add doc
        """

        # Initialisation
        x = physics.A_adjoint(y)  # New init

        for it in range(self.max_iter):

            x_prev = torch.clone(x)
            temp = physics.A(x) - y
            grad = physics.A_adjoint(temp)
            x_cur = x - self.gamma * grad
            x = self.denoiser(x_cur, denoise_level=self.denoise_level)

            if self.crit_conv is not None and (x_prev-x).norm() / (x.norm()+1e-03) < self.crit_conv:
                break

        return x
