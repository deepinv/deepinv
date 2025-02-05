import deepinv as dinv
from deepinv.models.flowunet import FlowUNet
from deepinv.sampling.pnpflow import PnPFlow
from deepinv.optim.data_fidelity import L2
from deepinv.utils.plotting import plot
from deepinv.utils.demo import load_url_image, get_image_url

import torch
mynet = FlowUNet(input_channels=3,
                 input_height=128, pretrained=True)


pnpflow = PnPFlow(mynet, data_fidelity=L2(),
                  verbose=True, max_iter=10, device='cpu')

url = get_image_url("69037.png")

x_true = load_url_image(url=url, img_size=256, device='cpu')

x = x_true.clone()
mask = torch.ones_like(x)
mask[:, :, 50:100, 50:100] = 0
mask[:, :, 80:130, 50:100] = 0

sigma_noise = 12.75 / 255.0  # noise level
physics = dinv.physics.Inpainting(
    mask=mask,
    tensor_size=x.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device='cpu',
)
y = physics(x)


imgs = [y, x_true]
plot(
    imgs,
    titles=["measurement", "ground-truth"],
)

x_hat = pnpflow.forward(y, physics)


imgs = [x_hat, x_true]
plot(
    imgs,
    titles=["reconstruction", "ground-truth"],
)
