import torch
import torch.nn as nn
from deepinv.diffops.models import drunet_testmode

class ProximalGradient(nn.Module):
    """
    TODO: add doc
    """

    def __init__(self, denoiser, denoise_level=None, max_iter=10, gamma=1, crit_conv=1e-4, test_mode=True, verbose=False):
        super(ProximalGradient, self).__init__()

        self.denoiser = denoiser
        self.denoise_level = denoise_level
        self.max_iter = max_iter
        self.gamma = gamma
        self.crit_conv = crit_conv
        self.test_mode = test_mode
        self.verbose = verbose

    def forward(self, y, physics):
        """
        TODO: add doc
        """

        with torch.set_grad_enabled(not self.test_mode):  # Enable grad at train time and disable it at test time

            # Initialisation
            x = physics.A_adjoint(y)*0  # New init

            for it in range(self.max_iter):

                x_prev = torch.clone(x)
                temp = physics.A(x) - y
                grad = physics.A_adjoint(temp)
                x_cur = x - self.gamma * grad
                x = self.denoise(x_cur)

                crit_cur = (x_prev-x).norm() / (x.norm()+1e-03) # For convergence analysis
                if self.verbose:
                    print(it, 'crit = ', crit_cur , '\r')

                if self.crit_conv is not None and crit_cur < self.crit_conv:
                    break

        return x

    def denoise(self, image):

        with torch.set_grad_enabled(not self.test_mode):  # Enable grad at train time and disable it at test time

            if 'UNetRes' in self.denoiser.__class__.__name__:
                denoised_image = drunet_testmode(self.denoiser, image)
            elif 'DnCNN' in self.denoiser.__class__.__name__:
                image[image>1] = 1
                denoised_image = self.denoiser(image, denoise_level=self.denoise_level)

        return denoised_image
