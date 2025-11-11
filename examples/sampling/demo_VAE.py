# %% [markdown]
# # VAE(Denoiser) — interactive demo
# Minimal encode/decode round-trip using a Diffusers VAE.
# Plots & saves are done via `dinv.utils.plot`.
#
# Steps:
# 1) Configure paths & options
# 2) Load/prepare image
# 3) Init VAE
# 4) Encode → Decode (round-trip) + metrics
# 5) (Optional) Decode a random latent

# %%
from __future__ import annotations
from pathlib import Path
import math
import random
import torch
import numpy as np
from PIL import Image
import deepinv as dinv
from deepinv.models import VAE

# %% [markdown]
# ## 1) Configure

# Fixed seed (CPU + GPU + NumPy + Python's RNG)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Determinism hints (may reduce performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
# Model / device / dtype
MODEL_ID = "stabilityai/sd-vae-ft-mse"  # or a local path
SUBFOLDER = None                        # e.g., "vae" if the VAE lives in a subfolder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Image I/O
IMAGE_PATH = Path("prior.png")  # e.g., Path("path/to/image.png"); if None, a random image is used
H, W = 512, 512
OUTDIR = Path("vae_demo_out"); OUTDIR.mkdir(parents=True, exist_ok=True)
FIGSIZE = 6  # inches for dinv.utils.plot

# Behavior
USE_TILING = False
USE_SLICING = False
SAMPLE_POSTERIOR = False  # True = sample z; False = use mean


# %% [markdown]
# ## 2) Load / prepare image (tensor in [0,1], shape (1,3,H,W))

# %%
if IMAGE_PATH is None:
    x = torch.rand((1, 3, H, W), device=DEVICE, dtype=DTYPE)
else:
    img = Image.open(IMAGE_PATH).convert("RGB")
    if (img.height, img.width) != (H, W):
        img = img.resize((W, H), Image.BICUBIC)
    arr = np.array(img) / 255.0
    x = torch.from_numpy(arr).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device=DEVICE, dtype=DTYPE).clamp(0.0, 1.0)

dinv.utils.plot(
    x.detach().cpu(),
    titles="Input",
    save_fn=str(OUTDIR / "input.png"),
    figsize=(FIGSIZE, FIGSIZE),
)


# %% [markdown]
# ## 3) Init VAE

# %%
vae = VAE.from_pretrained(
    MODEL_ID,
    subfolder=SUBFOLDER,
    device=DEVICE,
    torch_dtype=DTYPE,
    use_tiling=USE_TILING,
    use_slicing=USE_SLICING,
    sample_default=SAMPLE_POSTERIOR,
)
print(f"scaling_factor = {vae.scaling_factor:.6f}")


# %% [markdown]
# ## 4) Encode → Decode (round-trip) + metrics

# %%
with torch.no_grad():
    z = vae.encode(x, sample=SAMPLE_POSTERIOR)  # scaled latents
    x_rec = vae.decode(z)                       # [0,1]

# Metrics (MSE/PSNR)
mse = torch.mean((x.float() - x_rec.float()) ** 2).item()
psnr = float("inf") if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
print(f"MSE: {mse:.6f} | PSNR: {psnr:.2f} dB")

dinv.utils.plot(
    [x.detach().cpu(), x_rec.detach().cpu()],
    titles=["Input", "Reconstruction"],
    save_fn=str(OUTDIR / "reconstruction.png"),
    figsize=(FIGSIZE, FIGSIZE),
)

# %% [markdown]
# ## 5) Test a pre-trained AE wrapped in the VAE class

# %%
from deepinv.models import AutoEncoder

ckpt = torch.load("checkpoints_ae/ae_best.pt", map_location=DEVICE, weights_only=True)
in_shape = ckpt.get("in_shape")

# Recreate the AE with the SAME dims as training
model = AutoEncoder(
    dim_input=ckpt["dim_input"],
    dim_mid=ckpt["dim_mid"],
    dim_hid=ckpt["dim_hid"],
    residual=ckpt["residual"],
).to(DEVICE)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Wrap as a generic VAE: pass the *image* shape, not the flattened size
vae = VAE.from_generic(model, data_shape=in_shape, device=DEVICE)

img = Image.open(IMAGE_PATH).convert("RGB")
if (img.height, img.width) != in_shape[1:3]:
    img = img.resize((in_shape[2], in_shape[1]), Image.BICUBIC)
arr = np.array(img) / 255.0
x = torch.from_numpy(arr).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
x = x.to(device=DEVICE, dtype=DTYPE).clamp(0.0, 1.0)

# Encode / Decode with VAE wrapper over the AutoEncoder
with torch.no_grad():
    z = vae.encode(x)        # latent
    x_rec = vae.decode(z)    # (1,C,H,W)

# Metrics (MSE/PSNR)
mse = torch.mean((x.float() - x_rec.float()) ** 2).item()
psnr = float("inf") if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
print(f"MSE: {mse:.6f} | PSNR: {psnr:.2f} dB")

# Side-by-side panels: [Input | Reconstruction]
panel = torch.cat([x.detach().cpu(), x_rec.detach().cpu()], dim=-1)  # (1,C,H,2W)

dinv.utils.plot(
    panel,
    titles=[f"Input | AE recon"],
    save_fn=str(OUTDIR / "reconstruction.png"),
    figsize=(2 * FIGSIZE, FIGSIZE),
)

# %%
