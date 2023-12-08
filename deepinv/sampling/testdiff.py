import deepinv as dinv
import torch
import numpy as np
import matplotlib.pyplot as plt

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
x = 0.5 * torch.ones(1, 3, 128, 128)  # Define random 128x128 image
sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(
    mask=0.5,
    tensor_size=(3, 128, 128),
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma),
)

y = physics(x)  # measurements

denoiser = dinv.models.DRUNet(pretrained="download").to(device)

model = dinv.sampling.DDRM(
    denoiser=denoiser, sigmas=np.linspace(1, 0, 10), verbose=True
)
xhat = model(y, physics)

# plot x y and xhat
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(x[0].permute(1, 2, 0).cpu().numpy())
axs[0].set_title("x")
axs[1].imshow(y[0].permute(1, 2, 0).cpu().numpy())
axs[1].set_title("y")
axs[2].imshow(xhat[0].permute(1, 2, 0).cpu().numpy())
axs[2].set_title("xhat")
plt.show()
