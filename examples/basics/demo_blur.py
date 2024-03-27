r"""
Creating a forward operator.
====================================================================================================

Illustrating the possibilities offered to simulate blurs: stationary or space varying.

"""

import deepinv as dinv
import torch
from deepinv.utils.plotting import plot
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.physics.generator.blur import DiffractionBlurGenerator


device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = get_image_url("CBSD_0010.png")
x = load_url_image(url, grayscale=False).to(device)

x = torch.tensor(x, device=device, dtype=torch.float)
