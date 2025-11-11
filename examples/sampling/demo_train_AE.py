# %% [markdown]
# # demo_train_AE — Urban100HR (x → x) with deepinv AutoEncoder
#
# - Loads `Urban100HR` from `dinv.datasets` with simple transforms
# - Trains `deepinv.models.AutoEncoder` to reconstruct the input (MSE)
# - Tracks train/val loss and PSNR
# - Saves best checkpoint + comparison plots via `dinv.utils.plot`

# %% [markdown]
# ## 1) Imports & config

# %%
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from tqdm import tqdm

import deepinv as dinv
from deepinv.models import AutoEncoder  # <-- use AutoEncoder from deepinv.models

from deepinv.utils.demo import get_data_home

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurments"
ORGINAL_DATA_DIR = get_data_home() / "Urban100"

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

SAVE_DIR = Path("checkpoints_ae")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = SAVE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# data/loader
RESIZE = 256
CROP = 128
BATCH_SIZE = 16
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8

# training
EPOCHS = 20
LR = 1e-3

# model (matches deepinv AutoEncoder signature)
DIM_MID = 2048
DIM_HID = 256
RESIDUAL = True  # set False to disable residual connection

FIGSIZE = 3  # inches per image in dinv.utils.plot

# seeds
torch.manual_seed(SEED)

# %% [markdown]
# ## 2) Dataset & loaders

# %%
transform = Compose([ToTensor(), Resize(RESIZE), CenterCrop(CROP)])
dataset = dinv.datasets.Urban100HR(root=ORGINAL_DATA_DIR, download=True, transform=transform)

n_train = int(TRAIN_SPLIT * len(dataset))
n_val = len(dataset) - n_train
train_set, val_set = random_split(
    dataset, (n_train, n_val), generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
)

# %% [markdown]
# ## 3) Infer input dim and build the AutoEncoder

# %%
# get one batch to infer (C,H,W) and flatten size
with torch.no_grad():
    sample = next(iter(train_loader))
    # Urban100HR returns a tensor; if dict/tuple, get first tensor
    if isinstance(sample, dict) and "x" in sample:
        sample = sample["x"]
    elif isinstance(sample, (list, tuple)):
        sample = sample[0]
    in_shape = tuple(sample.shape[1:])  # (C,H,W)
    DIM_INPUT = int(torch.tensor(in_shape).prod().item())

print(f"[info] Input shape: {in_shape} -> dim_input={DIM_INPUT}")

# build model/optim/loss
model = AutoEncoder(
    dim_input=DIM_INPUT,
    dim_mid=DIM_MID,
    dim_hid=DIM_HID,
    residual=RESIDUAL,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# %% [markdown]
# ## 4) Train / validate + save best + plot comparisons

# %%
def _psnr(a, b, eps=1e-12):
    mse = torch.mean((a.float() - b.float()) ** 2)
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))

best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    running = 0.0
    for x in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        if isinstance(x, dict) and "x" in x: x = x["x"]
        elif isinstance(x, (list, tuple)): x = x[0]
        x = x.to(DEVICE)

        y = model(x)            # x -> x_hat
        loss = criterion(y, x)  # MSE recon

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)

    train_loss = running / len(train_loader.dataset)

    # ---- validate ----
    model.eval()
    val_sum, val_psnr = 0.0, 0.0
    vis_x, vis_y = None, None

    with torch.no_grad():
        for x in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
            if isinstance(x, dict) and "x" in x: x = x["x"]
            elif isinstance(x, (list, tuple)): x = x[0]
            x = x.to(DEVICE)

            y = model(x)
            loss = criterion(y, x)
            val_sum += loss.item() * x.size(0)
            val_psnr += _psnr(y.clamp(0, 1), x.clamp(0, 1)).item() * x.size(0)

            if vis_x is None:
                vis_x, vis_y = x.detach().cpu(), y.detach().cpu()

    val_loss = val_sum / len(val_loader.dataset)
    val_psnr = val_psnr / len(val_loader.dataset)
    print(f"[epoch {epoch}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_psnr={val_psnr:.2f} dB")

    # ---- save best ----
    if val_loss < best_val:
        best_val = val_loss
        ckpt_path = SAVE_DIR / "ae_best.pt"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "dim_input": DIM_INPUT,
                "dim_mid": DIM_MID,
                "dim_hid": DIM_HID,
                "residual": RESIDUAL,
                "in_shape": in_shape,
            },
            ckpt_path,
        )
        print(f"[save] best checkpoint -> {ckpt_path}")

    # ---- plot a few reconstructions ----
    if vis_x is not None:
        # vis_x, vis_y are (B, C, H, W) on CPU (from the val loop)
        N_PAIRS = min(2, vis_x.size(0))
        # idx = torch.randperm(vis_x.size(0))[:N_PAIRS]   # <- random two
        idx = torch.arange(N_PAIRS)                        # <- first two

        x_show = vis_x[idx].clamp(0, 1)  # (N, C, H, W)
        y_show = vis_y[idx].clamp(0, 1)  # (N, C, H, W)

        # build side-by-side panels: [x_i | AE(x_i)]
        panels = torch.stack(
            [torch.cat([x_show[i], y_show[i]], dim=-1) for i in range(N_PAIRS)],
            dim=0,  # (N, C, H, 2W)
        )

        titles = [f"x[{int(i)}] | AE(x)[{int(i)}]" for i in idx]

        # make them bigger: width scales with number of panels
        FIG_W, FIG_H = 6 * N_PAIRS, 6  # inches
        dinv.utils.plot(
            panels,
            titles=titles,
            save_fn=str(PLOT_DIR / f"epoch_{epoch:03d}_recon_2pairs.png"),
            figsize=(FIG_W, FIG_H),
        )


print("[done] Training complete.")

# %% [markdown]
# ## 5) Load best checkpoint

# %%
ckpt = torch.load("checkpoints_ae/ae_best.pt", map_location=DEVICE)

# Recreate the model with the same architecture
model = AutoEncoder(
    dim_input=ckpt["dim_input"],
    dim_mid=ckpt["dim_mid"],
    dim_hid=ckpt["dim_hid"],
    residual=ckpt["residual"],
).to(DEVICE)

# Load weights
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()
# %%
