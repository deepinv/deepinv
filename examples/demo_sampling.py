import deepinv as dinv
from deepinv.utils.plotting import plot_debug
import torch
import requests
from imageio.v2 import imread
from io import BytesIO
from pathlib import Path

# load image from the internet or from set3c dataset if no connection
try : 
    url = (
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
        "Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"
    )
    res = requests.get(url)
    x = imread(BytesIO(res.content)) / 255.0
    pretrained = "download_lipschitz"
except : 
    BASE_DIR = Path('..')
    ORIGINAL_DATA_DIR = BASE_DIR / 'datasets'
    im_path = ORIGINAL_DATA_DIR / 'set3c' / 'images/0/butterfly.png'
    x = imread(str(im_path)) / 255.0
    CKPT_DIR = BASE_DIR / 'checkpoints'
    pretrained = str(CKPT_DIR / 'dncnn_sigma2_lipschitz_color.pth')

x = torch.tensor(x, device=dinv.device, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
x = torch.nn.functional.interpolate(
    x, scale_factor=0.5
)  # reduce the image size for faster eval

# define forward operator
sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(mask=0.5, tensor_size=x.shape[1:], device=dinv.device)
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# load pretrained dncnn denoiser
model_spec = {
    "name": "dncnn",
    "args": {
        "device": dinv.device,
        "in_channels": 3,
        "out_channels": 3,
        "pretrained": pretrained,
    },
}

sigma_denoiser = 2 / 255
prior = dinv.models.ScoreDenoiser(
    model_spec=model_spec
)

# load Gaussian Likelihood
likelihood = dinv.optim.L2(sigma=sigma)

# choose MCMC sampling algorithm
regularization = 0.9
step_size = 0.01 * (sigma**2)
iterations = int(5e3)
f = dinv.sampling.ULA(
    prior=prior,
    data_fidelity=likelihood,
    max_iter=iterations,
    alpha=regularization,
    step_size=step_size,
    verbose=True,
    sigma=sigma_denoiser,
)

# generate measurements
y = physics(x)

# run algo
mean, var = f(y, physics)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_lin):.2f} dB")
print(f"Posterior mean PSNR: {dinv.utils.metric.cal_psnr(x, mean):.2f} dB")

# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [x_lin, x, mean, std / std.flatten().max(), error / error.flatten().max()]
plot_debug(
    imgs,
    titles=["measurement", "ground truth", "post. mean", "post. std", "abs. error"],
)
