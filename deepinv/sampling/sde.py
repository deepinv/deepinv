import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable
#from deepinv.utils.plotting import plot
from deepinv.utils.demo import load_url_image

class DiffusionSDE(nn.Module):
    def __init__(self, f: Callable, g: Callable, score: Callable, T: float = 1.0):
        super().__init__()
        self.f = f
        self.g = g
        self.score = score
        self.T = T

    def forward_sde(self, x: Tensor, num_steps: int) -> Tensor:
        x_new = x
        stepsize = 1.0 / num_steps
        for t in range(num_steps):
            dw      = torch.randn_like(x_new)
            f_dt    = self.f(x_new, t)
            g_dw    = self.g(t) * dw
            x_new   = x_new + stepsize*(f_dt + g_dw)
        return x_new

    def backward_sde(self, x: Tensor, t: Tensor) -> Tensor:
        pass

if __name__ == '__main__':
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

    url = (
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
        "Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"
    )
    x = load_url_image(url=url, img_size=32).to(device)

    DiffSDE = DiffusionSDE()

    
