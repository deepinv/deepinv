import deepinv as dinv
from deepinv.models.flowunet import FlowUNet
from deepinv.sampling.pnpflow import PnPFlow
from deepinv.optim.data_fidelity import L2
from deepinv.utils.plotting import plot
from deepinv.utils.demo import load_url_image, get_image_url

import torch
device = 'cuda'


mynet = FlowUNet(input_channels=3,
                 input_height=128, pretrained=True, device=device)

pnpflow = PnPFlow(mynet, data_fidelity=L2(),
                  verbose=True, max_iter=100, device=device, lr=1.0, lr_exp=0.5)


# url = get_image_url("celeba_example.jpg")
url = get_image_url("69037.png")

x_true = load_url_image(url=url, img_size=128,
                        resize_mode="resize", device=device)

x = x_true.clone()
mask = torch.ones_like(x)
mask[:, :, 44:84, 44:84] = 0

sigma_noise = 12.5 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    mask=mask,
    tensor_size=x.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)
y = physics(2*x - 1)
x_hat = pnpflow.forward(y, physics)


imgs = [y, x_true, (x_hat + 1.0)*0.5]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn='res_inpainting.png', save_dir='.'
)


#########################
pnpflow = PnPFlow(mynet, data_fidelity=L2(),
                  verbose=True, max_iter=100, device=device, lr=1.0, lr_exp=1.0)

url = get_image_url("celeba_example2.jpg")

x_true = load_url_image(url=url, img_size=128,
                        resize_mode="resize", device=device)

x = x_true.clone()

sigma_noise = 12.75 / 255.0  # noise level

physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:], filter=dinv.physics.blur.gaussian_blur(sigma=1.),
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device)
y = physics(2*x-1)

x_hat = pnpflow.forward(y, physics)

imgs = [y, x_true, (x_hat + 1.0)*0.5]
plot(
    imgs,
    titles=["measurement", "ground-truth", "reconstruction"],
    save_fn='res_blurfft.png', save_dir='.'
)
