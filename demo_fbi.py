"""Blind Poisson-Gaussian denoising with native DeepInverse models."""

from pathlib import Path

import torch
import deepinv as dinv

from deepinv.models import PGENet, FBINet

dinv.utils.disable_tex()
torch.manual_seed(0)
device = dinv.utils.get_device()
x = dinv.utils.load_example("butterfly.png", device=device)
physics = dinv.physics.Denoising(
    dinv.physics.PoissonGaussianNoise(sigma=0.02, gain=0.05, clip_positive=True),
    device=device,
)
y = physics(x)

pge_weights = Path(__file__).with_name(
    "211127_PGE_Net_RawRGB_fivek_alpha_0.05_beta_0.02_cropsize_200.w"
)
fbi_weights = Path(__file__).with_name(
    "211127_FBI_Net_RawRGB_fivek_alpha_0.05_beta_0.02_"
    "layers_x17_filters_x64_cropsize_220.w"
)
pge_state = torch.load(pge_weights, map_location=device, weights_only=True)
fbi_state = torch.load(fbi_weights, map_location=device, weights_only=True)
pge = dinv.models.PoissonGaussianEstimator(
    PGENet(square_output=True), eps=0.0, noise_map=True
).to(device)
fbi = FBINet().to(device)
pge.backbone_net.load_state_dict(pge_state)
fbi.load_state_dict(fbi_state)
pge.eval()
fbi.eval()
denoiser = dinv.models.AnscombeDenoiser(fbi)

with torch.no_grad():
    channels = y.squeeze(0).unsqueeze(1)  # RGB channels as independent 1-channel images
    params = pge(channels)
    sigma_maps, gain_maps =  params["sigma"], params["gain"]
    sigma, gain = sigma_maps.mean(), gain_maps.mean()
    estimate = denoiser(channels, sigma, gain).clamp(0, 1)
    estimate = estimate.squeeze(1).unsqueeze(0)

score = lambda z: (
    dinv.metric.PSNR()(z, x).mean().item(),
    dinv.metric.SSIM()(z, x).mean().item(),
)
noisy_score, estimate_score = map(score, (y, estimate))
gain_map = gain_maps.mean(0, keepdim=True)
sigma_map = sigma_maps.mean(0, keepdim=True)
norm = lambda z: (z - z.amin()) / (z.amax() - z.amin()).clamp_min(1e-8)
dinv.utils.plot(
    [x, y, norm(sigma_map), norm(gain_map), estimate],
    [
        "Clean",
        f"Noisy\nPSNR {noisy_score[0]:.2f} dB | SSIM {noisy_score[1]:.4f}",
        f"Native sigma map\nmean={sigma:.4f}",
        f"Native gain map\nmean={gain:.4f}",
        f"Native estimate\nPSNR {estimate_score[0]:.2f} dB | "
        f"SSIM {estimate_score[1]:.4f}",
    ],
    "out.png",
    rescale_mode="clip",
    show=False,
    close=True,
    dpi=200,
    figsize=(15, 3.5),
)
print(
    "model   gain/sigma       PSNR       SSIM\n"
    f"noisy   0.0500/0.0200    {noisy_score[0]:6.2f}    {noisy_score[1]:.4f}\n"
    f"native  {gain:.4f}/{sigma:.4f}    "
    f"{estimate_score[0]:6.2f}    {estimate_score[1]:.4f}"
)
