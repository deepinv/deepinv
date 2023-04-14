import deepinv as dinv
from deepinv.utils.plotting import plot_debug
import torch
import requests
from imageio.v2 import imread
from io import BytesIO

# load image from the internet
url = 'https://upload.wikimedia.org/wikipedia/commons/b/b4/' \
      'Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg'
res = requests.get(url)
x = imread(BytesIO(res.content))/255.
x = torch.tensor(x, device=dinv.device, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)
x = torch.nn.functional.interpolate(x, scale_factor=.5)  # reduce the image size for faster eval

# define forward operator
sigma = .1  # noise level
physics = dinv.physics.Inpainting(mask=.5, tensor_size=x.shape[1:], device=dinv.device)
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma)

# load pretrained dncnn denoiser
model_spec = {'name': 'dncnn', 'args': {'device': dinv.device, 'in_channels': 3, 'out_channels': 3,
                                        'pretrained': 'download_lipschitz'}}
prior = dinv.models.ScoreDenoiser(model_spec=model_spec, sigma_denoiser=2/255, device=dinv.device)

# load Gaussian Likelihood
likelihood = dinv.optim.L2(sigma=sigma)

# choose MCMC sampling algorithm
regularization = .9
step_size = .01*(sigma**2)
iterations = int(5e3)
f = dinv.sampling.ULA(prior=prior, data_fidelity=likelihood, max_iter=iterations,
                      alpha=regularization, step_size=step_size, verbose=True)

# generate measurements
y = physics(x)

# run algo
mean, var = f(y, physics)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# compute PSNR
print(f'Linear reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_lin):.2f} dB')
print(f'Posterior mean PSNR: {dinv.utils.metric.cal_psnr(x, mean):.2f} dB')

# plot results
error = (mean-x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [x_lin, x, mean, std/std.flatten().max(), error/error.flatten().max()]
plot_debug(imgs, titles=['measurement', 'ground truth', 'post. mean', 'post. std', 'abs. error'])