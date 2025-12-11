import torch
import deepinv as dinv
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

dataset = dinv.datasets.Urban100HR("data/Urban100", download=True, transform=transforms.ToTensor())

rng = torch.Generator()
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=25/255, rng=rng)).to(device)

model = dinv.models.DRUNet().to(device).eval()

psnr_fn = dinv.metric.PSNR(min_pixel=0.0, max_pixel=1.0).to(device)

psnrs = []
for k, x in enumerate(tqdm(dataset)):
    x = x.unsqueeze(0).to(device)
    y = physics(x, seed=k)

    with torch.no_grad():
        x_hat = model(y, physics.noise_model.sigma)

    psnr = psnr_fn(x_hat, x).item()
    psnrs.append(psnr)

psnr_avg, psnr_std = np.mean(psnrs), np.std(psnrs)

print(f"PSNR (dB): {psnr_avg:.2f} ± {psnr_std:.2f}")
