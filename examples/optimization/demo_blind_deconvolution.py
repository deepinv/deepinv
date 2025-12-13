# %%
import deepinv
from deepinv.utils import load_example
from deepinv.utils.plotting import plot
from deepinv.utils import get_freer_gpu
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

device = get_freer_gpu() if torch.cuda.is_available() else "cpu"


def richardson_lucy(
    y: torch.Tensor,
    x0: torch.Tensor,
    physics: deepinv.physics.LinearPhysics,
    steps: int,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
) -> torch.Tensor:
    """
    Performs Richardson-Lucy deconvolution on an observed image.

    Args:
        y (torch.Tensor): The observed image
        x0 (torch.Tensor): The initial estimate
        physics (deepinv.physics.LinearPhysics): The physics operator
        steps (int): Number of iterations
        verbose (bool): Whether to show progress bar
        keep_inter (bool): Whether to keep intermediate results

    Returns:
        torch.Tensor or tuple: The deconvolved image, and if keep_inter=True, list of intermediate results
    """
    xs = [x0.cpu().clone()] if keep_inter else None

    with torch.no_grad():
        recon = x0.clone()
        recon = recon.clamp(min=filter_epsilon)
        s = physics.A_adjoint(torch.ones_like(y))

        for step in tqdm(range(steps), desc="MLEM", disable=not verbose):
            recon = (recon / s) * physics.A_adjoint(
                y / physics.A(recon).clamp(min=filter_epsilon)
            )

            if keep_inter:
                xs.append(recon.cpu().clone())

        return (recon, xs) if keep_inter else recon


img_size = 256
x = load_example(
    "leaves.png",
    img_size=256,
    grayscale=False,
    device=device,
)

# Example with the blur physics and noise-free observations
kernel = deepinv.physics.blur.gaussian_blur(sigma=1.6)
physics = deepinv.physics.Blur(
    filter=kernel,
    padding="circular",
    device=device,
)

y = physics.forward(x)

x_mlem, xs_mlem = richardson_lucy(
    y=y,
    x0=y,
    physics=physics,
    steps=500,
    keep_inter=True,
)
psnr = deepinv.loss.metric.PSNR()

vmin = 0
vmax = 1

plot(
    [x, y, x_mlem],
    titles=["Ground Truth", "Blurry image", "Richardson-Lucy\nDeconvolution"],
    subtitles=[
        "PSNR (dB)",
        f"{psnr(x, y).item():.2f}",
        f"{psnr(x, x_mlem).item():.2f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)
x = x.cpu()
psnr_values = [psnr(x, x_iter).item() for x_iter in xs_mlem]
plt.plot(
    psnr_values,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Richardson-Lucy Deconvolution\nPSNR vs Iterations")
plt.show()
# %%
# Example with the blurFFT physics and noise-free observations
x = x.to(device)
physics = deepinv.physics.BlurFFT(
    img_size=(3, img_size, img_size),
    filter=kernel,
    device=device,
)

y = physics.forward(x)


x_mlem, xs_mlem = richardson_lucy(
    y=y,
    x0=y,
    physics=physics,
    steps=500,
    keep_inter=True,
)
psnr = deepinv.loss.metric.PSNR()

vmin = 0
vmax = 1

plot(
    [x, y, x_mlem],
    titles=["Ground Truth", "Blurry image", "Richardson-Lucy\nDeconvolution"],
    subtitles=[
        "PSNR (dB)",
        f"{psnr(x, y).item():.2f}",
        f"{psnr(x, x_mlem).item():.2f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)
x = x.cpu()
psnr_values = [psnr(x, x_iter).item() for x_iter in xs_mlem]
plt.plot(
    psnr_values,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Richardson-Lucy Deconvolution\nPSNR vs Iterations")
plt.show()

# %%
# Example with Poisson-corrupted observations
# Typical artifacts

x = x.to(device)
gain = 1 / 200
physics = deepinv.physics.Blur(
    filter=kernel,
    padding="circular",
    noise_model=deepinv.physics.noise.PoissonNoise(gain=gain),
    device=device,
)


y = physics.forward(x)


x_mlem, xs_mlem = richardson_lucy(
    y=y,
    x0=y,
    physics=physics,
    steps=500,
    keep_inter=True,
)
psnr = deepinv.loss.metric.PSNR()

vmin = 0
vmax = 1

plot(
    [x, y, x_mlem],
    titles=["Ground Truth", "Blurry image", "Richardson-Lucy\nDeconvolution"],
    subtitles=[
        "PSNR (dB)",
        f"{psnr(x, y).item():.2f}",
        f"{psnr(x, x_mlem).item():.2f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)
x = x.cpu()
psnr_values = [psnr(x, x_iter).item() for x_iter in xs_mlem]
plt.plot(
    psnr_values,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Richardson-Lucy Deconvolution\nPSNR vs Iterations")
plt.show()


# %%
# Blind setting
import torch.nn.functional as F


def _normalize_kernel(k: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Project kernel to the simplex: k >= 0 and sum_{i,j} k = 1 (per batch)."""
    k = k.clamp_min(0.0)
    z = k.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return k / z


def _get_padding_mode(physics, default: str = "circular") -> str:
    # deepinv blur physics typically has a `padding` attribute (e.g. "circular").
    # If not present, fall back to circular (good default for deblurring benchmarks).
    return getattr(physics, "padding", default)


@torch.no_grad()
def blind_richardson_lucy(
    y: torch.Tensor,
    x0: torch.Tensor,
    k0: torch.Tensor,
    physics: deepinv.physics.LinearPhysics,
    steps: int,
    x_steps: int = 1,
    k_steps: int = 1,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    normalize_kernel: bool = True,
):
    """
    Blind Richardson–Lucy (alternating MLEM) for deblurring.

    Assumes a Poisson data model with a convolutional forward operator:
        y ~ Poisson( A_k(x) )
    where A_k is blur with kernel k.

    The algorithm alternates:
      - kernel MLEM update (k fixed-size, nonnegative, optionally normalized)
      - image MLEM update (standard Richardson–Lucy using physics.A / physics.A_adjoint)

    Parameters
    ----------
    y : torch.Tensor
        Measurements, shape (B, C, H, W).
    x0 : torch.Tensor
        Initial latent image, shape (B, C, H, W).
    k0 : torch.Tensor
        Initial kernel, shape (B, 1, hk, wk) or (1, 1, hk, wk).
        (If you pass (1,1,hk,wk), it will be broadcast across batch.)
    physics : deepinv.physics.LinearPhysics
        Blur physics (e.g. Blur/BlurFFT) supporting update(filter=...),
        and methods A(.) and A_adjoint(.).
    steps : int
        Number of outer alternations.
    x_steps : int
        Number of inner RL updates for x per outer iteration.
    k_steps : int
        Number of inner RL updates for k per outer iteration.
    verbose : bool
        Progress bar.
    keep_inter : bool
        If True, returns (x, k, xs, ks).
    filter_epsilon : float
        Numerical stability clamp (avoid division by 0).
    normalize_kernel : bool
        If True, enforces sum(k)=1 after each kernel update.

    Returns
    -------
    x_hat : torch.Tensor
        Estimated latent image.
    k_hat : torch.Tensor
        Estimated blur kernel (B,1,hk,wk).
    (optional) xs, ks : lists of torch.Tensor
        Intermediate iterates on CPU if keep_inter=True.
    """
    # Clone / stabilize
    x = x0.clone().clamp_min(filter_epsilon)

    # Broadcast kernel if needed
    k = k0.clone()
    if k.dim() != 4:
        raise ValueError("k0 must be a 4D tensor shaped (B or 1, 1, hk, wk).")
    if k.shape[0] == 1 and x.shape[0] > 1:
        k = k.expand(x.shape[0], -1, -1, -1).contiguous()
    k = k.to(dtype=x.dtype, device=x.device)

    if normalize_kernel:
        k = _normalize_kernel(k, eps=filter_epsilon)
    else:
        k = k.clamp_min(0.0)

    # Intermediate storage
    xs = [x.detach().cpu().clone()] if keep_inter else None
    ks = [k.detach().cpu().clone()] if keep_inter else None

    # Padding mode for the kernel update convolutions
    pad_mode = _get_padding_mode(physics, default="circular")

    # Precompute shapes
    B, C, H, W = y.shape
    hk, wk = k.shape[-2], k.shape[-1]

    # Pad sizes for "full" correlation then center-crop to hk x wk
    # We use large padding so that conv2d returns a big map from which we crop the center patch.
    # This matches your PyTorch implementation strategy.
    pad_k = (W // 2, W // 2, H // 2, H // 2)  # (left,right,top,bottom)

    for _ in tqdm(range(steps), desc="Blind RL", disable=not verbose):

        # -------------------------
        # (1) Kernel update (k|x)
        # -------------------------
        # Use luminance proxy so that one kernel explains all channels.
        xL = x.sum(dim=1, keepdim=True)  # (B,1,H,W)

        # flip for correlation
        xL_T = torch.flip(xL, dims=[-2, -1])  # (B,1,H,W)

        for _ in range(k_steps):
            # Update physics with current kernel estimate
            # (This is the deepinv way to pass new operator params.)
            if hasattr(physics, "update"):
                physics.update(filter=k)
            else:
                # Fallback if update is not available
                setattr(physics, "filter", k)

            y_hat = physics.A(x).clamp_min(filter_epsilon)

            # ratio: y / (A_k x)
            ratio = (y / y_hat).clamp_min(0.0)
            ratioL = ratio.mean(dim=1, keepdim=True)  # (B,1,H,W)

            onesL = torch.ones_like(ratioL)

            # We cannot directly do a per-sample kernel correlation in one conv2d call
            # (batch kernels are not supported). We loop over batch for correctness.
            k_new = []
            for b in range(B):
                rb = ratioL[b : b + 1]  # (1,1,H,W)
                xbT = xL_T[b : b + 1]  # (1,1,H,W)

                num_map = F.conv2d(F.pad(rb, pad_k, mode=pad_mode), xbT)
                den_map = F.conv2d(F.pad(onesL[b : b + 1], pad_k, mode=pad_mode), xbT)

                # center crop to (hk, wk)
                cy, cx = num_map.shape[-2] // 2, num_map.shape[-1] // 2
                num = num_map[
                    :,
                    :,
                    cy - hk // 2 : cy + hk // 2 + 1,
                    cx - wk // 2 : cx + wk // 2 + 1,
                ]
                den = den_map[
                    :,
                    :,
                    cy - hk // 2 : cy + hk // 2 + 1,
                    cx - wk // 2 : cx + wk // 2 + 1,
                ].clamp_min(filter_epsilon)

                kb = k[b : b + 1] * (num / den)
                kb = kb.clamp_min(0.0)
                if normalize_kernel:
                    kb = _normalize_kernel(kb, eps=filter_epsilon)
                k_new.append(kb)

            k = torch.cat(k_new, dim=0)

        # -------------------------
        # (2) Image update (x|k)
        # -------------------------
        physics.update_parameters(filter=k)
        s = physics.A_adjoint(torch.ones_like(y)).clamp_min(filter_epsilon)

        for _ in range(x_steps):
            Ax = physics.A(x).clamp_min(filter_epsilon)
            x = (x / s) * physics.A_adjoint(y / Ax)
            x = x.clamp_min(filter_epsilon)

        if keep_inter:
            xs.append(x.detach().cpu().clone())
            ks.append(k.detach().cpu().clone())

    if keep_inter:
        return x, k, xs, ks
    return x, k


img_size = 128
x = load_example(
    "SheppLogan.png",
    img_size=img_size,
    grayscale=True,
    device=device,
    resize_mode="resize",
)
# True physics (unknown kernel in blind setting)
kernel_true = deepinv.physics.blur.gaussian_blur(sigma=1.6)
physics = deepinv.physics.Blur(filter=kernel_true, padding="circular", device=device)
# Broken ?
# physics = deepinv.physics.BlurFFT(
#     img_size=(1, 128, 128), filter=kernel_true, padding="circular", device=device
# )

y = physics(x)

# Initial guesses
k0 = torch.ones((1, 1, 33, 33))

x_hat, k_hat = blind_richardson_lucy(
    y=y,
    x0=y,
    k0=k0,
    physics=physics,
    steps=200,
    x_steps=1,
    k_steps=1,
    verbose=True,
    keep_inter=False,
)

# %%
plot(
    [x, y, x_hat],
    titles=["Ground Truth", "Blurry image", "Blind Richardson-Lucy\nDeconvolution"],
    subtitles=[
        "PSNR (dB)",
        f"{psnr(x, y).item():.2f}",
        f"{psnr(x, x_hat).item():.2f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)

plot(
    [kernel_true, k_hat],
    titles=["True kernel", "Estimated kernel"],
    figsize=(6, 4),
)

# %%
