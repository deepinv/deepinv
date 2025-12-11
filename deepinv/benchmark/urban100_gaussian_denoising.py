from __future__ import annotations
import torch
import torch.utils.data
import deepinv as dinv
import torchvision.transforms as transforms
import numpy as np
from tqdm.auto import tqdm
from typing import Any
import pandas as pd  # noqa: TID253


class Benchmark:
    r"""
    Benchmark for Gaussian Denoising on Urban100 dataset

    .. note::

        The noise standard deviation is set to 25/255 for images normalized between 0 and 1.
    """

    @staticmethod
    def run(
        model: dinv.models.Denoiser,
        *,
        device: torch.device | str = torch.device("cpu"),
    ) -> Any:
        """Run the benchmark on the given model"""
        dataset = dinv.datasets.Urban100HR(
            "data/Urban100", download=True, transform=transforms.ToTensor()
        )

        rng = torch.Generator(device)
        physics = dinv.physics.Denoising(
            dinv.physics.GaussianNoise(sigma=25 / 255, rng=rng)
        ).to(device)

        psnr_fn = dinv.metric.PSNR(min_pixel=0.0, max_pixel=1.0).to(device)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, prefetch_factor=1
        )

        psnrs = []
        model = model.to(device).eval()
        for k, x in enumerate(tqdm(dataloader)):
            x = x.to(device)
            y = physics(x, seed=k)

            with torch.no_grad():
                x_hat = model(y, physics.noise_model.sigma)

            # Clip and quantize
            x_hat = x_hat.mul(255.0).round().div(255.0).clamp(0.0, 1.0)
            x = x.mul(255.0).round().div(255.0).clamp(0.0, 1.0)

            psnr = psnr_fn(x_hat, x).item()
            psnrs.append(psnr)
            if k >= 1:
                break

        return np.mean(psnrs), np.std(psnrs)


benchmark = Benchmark()
models = [dinv.models.DRUNet(), dinv.models.Restormer()]

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

rows = []
for model in models:
    model_name = type(model).__name__
    psnr_avg, psnr_std = benchmark.run(model, device=device)
    rows.append({"model_name": model_name, "psnr_avg": psnr_avg, "psnr_std": psnr_std})

df = pd.DataFrame(rows)
print(df)
