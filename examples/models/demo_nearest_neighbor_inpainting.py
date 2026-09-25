import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils import load_example

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

x = load_example("CBSD_0010.png", grayscale=False).to(device)

x = torch.tensor(x, device=device, dtype=torch.float)
x = torch.nn.functional.interpolate(x, size=(64, 64))

sigma = 0.1  # noise level
physics = dinv.physics.Inpainting(
    mask=0.5,
    img_size=x.shape[1:],
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    device=device,
)

y = physics(x)

# nearest neighbor (k-NN) inpainting
import torch_geometric
from torch_geometric.nn import knn

mask = physics.mask

# get a 2D boolean mask (H, W)
if mask.dim() == 4:      # (B, 1, H, W) or (B, C, H, W)
    mask_2d = mask[0, 0] > 0
elif mask.dim() == 3:    # (1, H, W) or (C, H, W)
    mask_2d = mask[0] > 0
else:                    # (H, W)
    mask_2d = mask > 0

H, W = mask_2d.shape

u = torch.arange(H, device=device)
v = torch.arange(W, device=device)
U, V = torch.meshgrid(u, v, indexing="ij")

# coordinates of known / unknown pixels
known_coords = torch.stack([U[mask_2d], V[mask_2d]], dim=-1).float()
unknown_coords = torch.stack([U[~mask_2d], V[~mask_2d]], dim=-1).float()

k = 7  # for example, any k >= 2

# knn returns indices into unknown_coords (row) and known_coords (col)
assign_index = knn(known_coords, unknown_coords, k=k)
y_idx, x_idx = assign_index  # y_idx: unknown indices, x_idx: known indices

num_unknown = unknown_coords.shape[0]

# reshape so each unknown pixel has k neighbors
x_idx = x_idx.view(num_unknown, k)   # (num_unknown, k)
y_idx = y_idx.view(num_unknown, k)   # (num_unknown, k), but each row is constant

# flat indices of known / unknown pixels
known_flat = torch.nonzero(mask_2d.view(-1), as_tuple=False).squeeze(1)     # (num_known,)
unknown_flat = torch.nonzero(~mask_2d.view(-1), as_tuple=False).squeeze(1)  # (num_unknown,)

# neighbors' flat indices for each unknown pixel
neighbors_flat = known_flat[x_idx]  # (num_unknown, k)

B, C, H, W = y.shape
HW = H * W

# flatten image over spatial dims
y_flat = y.view(B * C, HW)  # (B*C, H*W)

# gather k neighbor values for each unknown pixel: (B*C, num_unknown, k)
neighbors_vals = y_flat[:, neighbors_flat]

# average over neighbors: (B*C, num_unknown)
neighbors_mean = neighbors_vals.mean(dim=-1)

# fill unknown pixels
u_nn_flat = y_flat.clone()
u_nn_flat[:, unknown_flat] = neighbors_mean

# reshape back to (B, C, H, W)
u_nn = u_nn_flat.view(B, C, H, W)

# plot results
psnr_fn = dinv.metric.PSNR()
psnr_y = psnr_fn(y, x).item()
psnr_u = psnr_fn(u_nn, x).item()
plot(
    [x, y, u_nn],
    titles=[
        "signal",
        f"measurement ({psnr_y:.1f} dB)",
        f"{k}-NN inpaint ({psnr_u:.1f} dB)",
    ],
)
