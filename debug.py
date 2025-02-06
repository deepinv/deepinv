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
                  verbose=True, max_iter=100, device=device, lr=1.0, lr_exp=0.6)


# from PIL import Image
# ori_img = Image.open('example.jpg')
# import torchvision.transforms as v2


# # Define the transformation
# transform = v2.Compose([
#     v2.ToTensor(),
#     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Apply the transformation
# x_true = transform(ori_img).unsqueeze(0).to(device)
url = get_image_url("celeba_example.jpg")

x_true = load_url_image(url=url, img_size=128,
                        resize_mode="resize", device=device)

x = x_true.clone()
mask = torch.ones_like(x)
mask[:, :, 32:96, 32:96] = 0

sigma_noise = 12.5 / 255.0  # noise level

physics = dinv.physics.Inpainting(
    mask=mask,
    tensor_size=x.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
    device=device,
)
y = physics(x)


imgs = [y, x_true]
plot(
    imgs,
    titles=["measurement", "ground-truth"],  save_fn='degrad.png', save_dir='.'
)

x_hat = pnpflow.forward(y, physics)


imgs = [(x_hat + 1.0)*0.5, x_true]
plot(
    imgs,
    titles=["reconstruction", "ground-truth"], save_fn='res.png', save_dir='.'
)


pnpflow = PnPFlow(mynet, data_fidelity=L2(),
                  verbose=True, max_iter=100, device=device, lr=1.0, lr_exp=0.01)

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


imgs = [y, x_true]
plot(
    imgs,
    titles=["measurement", "ground-truth"],  save_fn='degrad2.png', save_dir='.'
)

x_hat = pnpflow.forward(y, physics)


imgs = [(x_hat + 1.0)*0.5, x_true]
plot(
    imgs,
    titles=["reconstruction", "ground-truth"], save_fn='res2.png', save_dir='.'
)
