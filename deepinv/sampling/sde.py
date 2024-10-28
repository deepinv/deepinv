import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable

class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        score: Callable = None,
        T: float = 1.0,
    ):
        super().__init__()
        self.f = f
        self.g = g
        self.score = score
        self.T = T

    def forward_sde(self, x: Tensor, num_steps: int) -> Tensor:
        x_new = x
        stepsize = self.T / num_steps
        for k in range(num_steps):
            t = stepsize * k
            dw      = torch.randn_like(x_new)
            f_dt    = self.f(x_new, t)
            g_dw    = self.g(t) * dw
            x_new   = x_new + stepsize*(f_dt + g_dw)
        return x_new

    def prior_sampling(self, batch_size: int = 1) -> Tensor:
        # return torch.randn((batch_size, ))
        pass
  
    def backward_sde(
        self, x: Tensor, num_steps: int = 1000, alpha: float = 1.0
    ) -> Tensor:
        dt = self.T / num_steps
        t = 0
        for i in range(num_steps):
            rt = self.T - t
            g = self.g(x, rt)
            drift = self.f(x, rt) - (1 + alpha**2) * g**2 * self.score(x, rt)
            diffusion = alpha * g

if __name__ == '__main__':
    import deepinv as dinv
    
    from deepinv.utils.plotting import plot
    from deepinv.utils.demo import load_url_image
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    url = (
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
        "Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"
    )
    x = load_url_image(url=url, img_size=32).to(device)

    DiffSDE = DiffusionSDE()

    x_for = DiffSDE.forward_sde(x, 10)



  