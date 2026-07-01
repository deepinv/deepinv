# %%
import deepinv
from deepinv.utils import load_example
from deepinv.utils.plotting import plot
from deepinv.utils import get_freer_gpu
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

device = get_freer_gpu() if torch.cuda.is_available() else "cpu"
psnr = deepinv.loss.metric.PSNR()

img_size = 128
x = load_example(
    "leaves.png",
    img_size=img_size,
    grayscale=False,
    device=device,
)
mae = deepinv.metric.MAE()
vmin = 0
vmax = 1

# Example with the blur physics and noise-free observations
kernel = deepinv.physics.functional.blur.gaussian_blur(psf_size=(25, 25), sigma=1.6)
physics = deepinv.physics.Blur(
    filter=kernel,
    padding="circular",
    device=device,
)

y = physics(x)
gain = 1 / 100
mlem = deepinv.optim.MLEM(
    data_fidelity=deepinv.optim.PoissonLikelihood(gain=gain),
    prior=None,
    max_iter=500,
)
x_mlem = mlem(y=y, physics=physics)
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
    vmin=0,
    vmax=1,
)
# %%
# Example with the blurFFT physics and noise-free observations
x = x.to(device)
physics = deepinv.physics.BlurFFT(
    img_size=(3, img_size, img_size),
    filter=kernel,
    device=device,
)

y = physics(x)

x_mlem = mlem(y=y, physics=physics)
psnr = deepinv.loss.metric.PSNR()

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
    vmin=0,
    vmax=1,
)


# %%
# Example with Poisson-corrupted observations
# Typical artifacts

x = x.to(device)
gain = 1 / 100
physics = deepinv.physics.Blur(
    filter=kernel,
    padding="circular",
    noise_model=deepinv.physics.noise.PoissonNoise(gain=gain),
    device=device,
)


y = physics(x)


x_mlem = mlem(y=y, physics=physics)
psnr = deepinv.loss.metric.PSNR()

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
    vmin=0,
    vmax=1,
)

# %%
# Algorithm definition


def _normalize_kernel(k: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Project kernel to the simplex: k >= 0 and sum_{i,j} k = 1 (per batch)."""
    k = k.clamp_min(0.0)
    z = k.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return k / z


def _circular_patches(x: torch.Tensor, kernel_size: tuple[int, int]) -> torch.Tensor:
    h, w = kernel_size
    ph, pw = h // 2, w // 2
    ih, iw = (h - 1) % 2, (w - 1) % 2
    x_pad = torch.nn.functional.pad(x, (pw - iw, pw, ph - ih, ph), mode="circular")
    patches = torch.nn.functional.unfold(x_pad, kernel_size=kernel_size)
    return patches.view(x.shape[0], x.shape[1], h * w, x.shape[-2] * x.shape[-1])


def _kernel_forward_from_patches(
    patches: torch.Tensor, k: torch.Tensor, out_shape: tuple[int, int, int, int]
) -> torch.Tensor:
    B, C, H, W = out_shape
    hk, wk = k.shape[-2:]
    k_flip = k.flip(-2, -1)
    if k_flip.shape[1] == 1 and C > 1:
        k_flip = k_flip.expand(-1, C, -1, -1)
    return (patches * k_flip.reshape(B, C, hk * wk, 1)).sum(dim=2).view(B, C, H, W)


def _kernel_adjoint_from_patches(
    patches: torch.Tensor, y: torch.Tensor, kernel_size: tuple[int, int]
) -> torch.Tensor:
    hk, wk = kernel_size
    y_flat = y.reshape(y.shape[0], y.shape[1], 1, -1)
    return (
        (patches * y_flat).sum(dim=-1).view(y.shape[0], y.shape[1], hk, wk).flip(-2, -1)
    )


@torch.no_grad()
def blind_richardson_lucy(
    y: torch.Tensor,
    x0: torch.Tensor,
    k0: torch.Tensor,
    steps: int,
    x_steps: int = 1,
    k_steps: int = 1,
    x_prior: deepinv.optim.Prior = deepinv.optim.ZeroPrior(),
    k_prior: deepinv.optim.Prior = deepinv.optim.ZeroPrior(),
    x_prior_weight: float = 0.0,
    k_prior_weight: float = 0.0,
    verbose: bool = True,
    keep_inter: bool = False,
    filter_epsilon: float = 1e-20,
    normalize_kernel: bool = True,
    fft: bool = True,
):
    """
    Blind Richardson-Lucy (alternating MLEM) for deblurring.

    Assumes a Poisson data model with a convolutional forward operator:
        y ~ Poisson( A_k(x) )
    where A_k is blur with kernel k.

    The algorithm alternates:
      - kernel MLEM update (k fixed-size, nonnegative, optionally normalized)
      - image MLEM update (standard Richardson-Lucy using physics.A / physics.A_adjoint)

    Parameters
    ----------
    y : torch.Tensor
        Measurements, shape (B, C, H, W).
    x0 : torch.Tensor
        Initial latent image, shape (B, C, H, W).
    k0 : torch.Tensor
        Initial kernel, shape (B, 1, hk, wk) or (1, 1, hk, wk).
        (If you pass (1,1,hk,wk), it will be broadcast across batch.)
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
    fft : bool
        If True, use BlurFFT for the image update. The kernel update assumes
        circular padding and works directly on the kernel support.

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

    # Precompute shapes
    B_im, C_im, H_im, W_im = y.shape
    H_k, W_k = k.shape[-2], k.shape[-1]

    def _expand_kernel_channels(k_in: torch.Tensor) -> torch.Tensor:
        if k_in.shape[1] == 1 and C_im > 1:
            return k_in.expand(-1, C_im, -1, -1)
        return k_in

    ones_y = torch.ones_like(y)

    if fft:
        physics = deepinv.physics.BlurFFT(
            img_size=(C_im, H_im, W_im),
            filter=_expand_kernel_channels(k),
            device=x.device,
        )

    else:
        physics = deepinv.physics.Blur(
            filter=_expand_kernel_channels(k),
            padding="circular",
            device=x.device,
        )

    for step in tqdm(range(steps), desc="Blind RL", disable=not verbose):

        # Kernel updates
        x_patches = _circular_patches(x, (H_k, W_k))
        s_k = _kernel_adjoint_from_patches(x_patches, ones_y, (H_k, W_k)).mean(
            dim=1, keepdim=True
        )
        s_k = s_k.clamp_min(filter_epsilon)

        for _ in range(k_steps):
            ratio = y / _kernel_forward_from_patches(
                x_patches, k, (B_im, C_im, H_im, W_im)
            ).clamp_min(filter_epsilon)
            num = _kernel_adjoint_from_patches(x_patches, ratio, (H_k, W_k)).mean(
                dim=1, keepdim=True
            )

            denom_k = (s_k + k_prior_weight * k_prior.grad(k)).clamp_min(filter_epsilon)
            k = (k / denom_k) * num
            k = k.clamp_min(0.0)
            if normalize_kernel:
                k = _normalize_kernel(k, eps=filter_epsilon)

        # Image updates
        physics.update_parameters(filter=_expand_kernel_channels(k))
        s = physics.A_adjoint(ones_y).clamp_min(filter_epsilon)

        for _ in range(x_steps):
            if isinstance(x_prior, deepinv.optim.TVPrior):
                grad_x = x_prior.nabla(x)
                denom_x = (
                    s
                    + x_prior_weight
                    * x_prior.nabla_adjoint(
                        grad_x / torch.abs(grad_x).clamp_min(filter_epsilon)
                    )
                ).clamp_min(filter_epsilon)
                x = (
                    x
                    / denom_x
                    * physics.A_adjoint(y / physics.A(x).clamp_min(filter_epsilon))
                )
            else:
                denom_x = (s + x_prior_weight * x_prior.grad(x)).clamp_min(
                    filter_epsilon
                )
                x = (x / denom_x) * physics.A_adjoint(
                    y / physics.A(x).clamp_min(filter_epsilon)
                )
            x = x.clamp_min(filter_epsilon)

        if keep_inter:
            xs.append(x.detach().cpu().clone())
            ks.append(k.detach().cpu().clone())

    if keep_inter:
        return x, k, xs, ks
    return x, k


# %%
# Noiseless grayscale example

img_size = 127
x = load_example(
    "SheppLogan.png",
    img_size=img_size,
    grayscale=True,
    device=device,
    resize_mode="resize",
)


# True physics (unknown kernel in blind setting)
kernel_true = deepinv.physics.functional.blur.gaussian_blur(
    psf_size=(25, 25), sigma=1.6
)
# kernel_true = torch.nn.functional.pad(
#     kernel_true,
#     (0, 1, 0, 1),
# )
physics = deepinv.physics.Blur(filter=kernel_true, padding="circular", device=device)
# physics = deepinv.physics.BlurFFT(
#     img_size=(1, img_size, img_size),
#     filter=kernel_true,
#     device=device,
# )

y = physics(x)

# Initial guesses
k0 = torch.ones_like(kernel_true)
x0 = y.clone()

max_iter = 20
x_hat, k_hat, xs, ks = blind_richardson_lucy(
    y=y,
    x0=x0,
    k0=k0,
    steps=max_iter,
    x_steps=1,
    k_steps=2,
    verbose=True,
    keep_inter=True,
    fft=False,
    normalize_kernel=True,
)
plot(
    [x, y, x_hat, kernel_true, k_hat],
    titles=[
        "Ground Truth",
        "Blurry image",
        "Blind Richardson-Lucy\nDeconvolution",
        "True kernel",
        "Estimated kernel",
    ],
    subtitles=[
        "PSNR (dB)",
        f"{psnr(x, y).item():.2f}",
        f"{psnr(x, x_hat).item():.2f}",
        "MAE",
        f"{mae(kernel_true, k_hat.cpu()).item():.4f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)

psnr_img = [psnr(x.cpu(), x_iter).item() for x_iter in xs]
mae_kernel = [mae(kernel_true.cpu(), k_iter).item() for k_iter in ks]

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(
    psnr_img,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Blind Richardson-Lucy Deconvolution\nImage PSNR vs Iterations")
plt.subplot(1, 2, 2)
plt.plot(
    mae_kernel,
)
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.title("Blind Richardson-Lucy Deconvolution\nKernel MAE vs Iterations")
plt.show()

# %%
# Noiseless RGB example

img_size = 127
x = load_example(
    "butterfly.png",
    img_size=img_size,
    grayscale=False,
    device=device,
    resize_mode="resize",
)
# True physics (unknown kernel in blind setting)
kernel_true = deepinv.physics.functional.blur.gaussian_blur(
    psf_size=(25, 25), sigma=1.6
)
physics = deepinv.physics.Blur(
    filter=kernel_true,
    padding="circular",
    # noise_model=deepinv.physics.PoissonNoise(gain=1 / 100),
    device=device,
)

y = physics(x)

# Initial guesses
k0 = torch.ones_like(kernel_true)
max_iter = 50
x_hat, k_hat, xs, ks = blind_richardson_lucy(
    y=y,
    x0=y,
    k0=k0,
    steps=max_iter,
    x_steps=1,
    k_steps=2,
    verbose=True,
    keep_inter=True,
    fft=False,
    normalize_kernel=True,
)

plot(
    [x, y, x_hat, kernel_true, k_hat],
    titles=[
        "Ground Truth",
        "Blurry image",
        "Blind Richardson-Lucy\nDeconvolution",
        "True kernel",
        "Estimated kernel",
    ],
    subtitles=[
        "PSNR (dB):",
        f"{psnr(x, y).item():.2f} dB",
        f"{psnr(x, x_hat).item():.2f} dB",
        "MAE:",
        f"{mae(kernel_true, k_hat.cpu()).item():.4f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)

psnr_img = [psnr(x.cpu(), x_iter).item() for x_iter in xs]
mae_kernel = [mae(kernel_true.cpu(), k_iter).item() for k_iter in ks]

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(
    psnr_img,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Blind Richardson-Lucy Deconvolution\nImage PSNR vs Iterations")
plt.subplot(1, 2, 2)
plt.plot(
    mae_kernel,
)
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.title("Blind Richardson-Lucy Deconvolution\nKernel MAE vs Iterations")
plt.show()

# %%
# Noisy grayscale example
img_size = 127
x = load_example(
    "SheppLogan.png",
    img_size=img_size,
    grayscale=True,
    device=device,
    resize_mode="resize",
)


# True physics (unknown kernel in blind setting)
kernel_true = deepinv.physics.functional.blur.gaussian_blur(
    psf_size=(25, 25), sigma=1.6
)

physics = deepinv.physics.BlurFFT(
    img_size=(1, img_size, img_size),
    noise_model=deepinv.physics.noise.PoissonNoise(gain=1 / 200, clip_positive=True),
    filter=kernel_true,
    device=device,
)

y = physics(x)

# Initial guesses
k0 = torch.ones_like(kernel_true)
x0 = y.clone()

max_iter = 20
x_hat, k_hat, xs, ks = blind_richardson_lucy(
    y=y,
    x0=x0,
    k0=k0,
    steps=max_iter,
    x_steps=1,
    k_steps=1,
    verbose=True,
    keep_inter=True,
    fft=False,
    normalize_kernel=True,
)
plot(
    [x, y, x_hat, kernel_true, k_hat],
    titles=[
        "Ground Truth",
        "Blurry image",
        "Blind Richardson-Lucy\nDeconvolution",
        "True kernel",
        "Estimated kernel",
    ],
    subtitles=[
        "PSNR (dB)",
        f"{psnr(x, y).item():.2f}",
        f"{psnr(x, x_hat).item():.2f}",
        "MAE",
        f"{mae(kernel_true, k_hat.cpu()).item():.4f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)

psnr_img = [psnr(x.cpu(), x_iter).item() for x_iter in xs]
mae_kernel = [mae(kernel_true.cpu(), k_iter).item() for k_iter in ks]

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(
    psnr_img,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Blind Richardson-Lucy Deconvolution\nImage PSNR vs Iterations")
plt.subplot(1, 2, 2)
plt.plot(
    mae_kernel,
)
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.title("Blind Richardson-Lucy Deconvolution\nKernel MAE vs Iterations")
plt.show()

# %%
# Noisy RGB example

img_size = 127
x = load_example(
    "butterfly.png",
    img_size=img_size,
    grayscale=False,
    device=device,
    resize_mode="resize",
)
# True physics (unknown kernel in blind setting)
kernel_true = deepinv.physics.functional.blur.gaussian_blur(
    psf_size=(25, 25), sigma=1.6
)
physics = deepinv.physics.Blur(
    filter=kernel_true,
    padding="circular",
    noise_model=deepinv.physics.PoissonNoise(gain=1 / 200, clip_positive=True),
    device=device,
)

y = physics(x)

# Initial guesses
k0 = torch.ones_like(kernel_true)
max_iter = 50
x_hat, k_hat, xs, ks = blind_richardson_lucy(
    y=y,
    x0=y,
    k0=k0,
    steps=max_iter,
    x_steps=1,
    k_steps=1,
    verbose=True,
    keep_inter=True,
    fft=False,
    normalize_kernel=True,
)

plot(
    [x, y, x_hat, kernel_true, k_hat],
    titles=[
        "Ground Truth",
        "Blurry image",
        "Blind Richardson-Lucy\nDeconvolution",
        "True kernel",
        "Estimated kernel",
    ],
    subtitles=[
        "PSNR (dB):",
        f"{psnr(x, y).item():.2f} dB",
        f"{psnr(x, x_hat).item():.2f} dB",
        "MAE:",
        f"{mae(kernel_true, k_hat.cpu()).item():.4f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)

psnr_img = [psnr(x.cpu(), x_iter).item() for x_iter in xs]
mae_kernel = [mae(kernel_true.cpu(), k_iter).item() for k_iter in ks]

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(
    psnr_img,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Blind Richardson-Lucy Deconvolution\nImage PSNR vs Iterations")
plt.subplot(1, 2, 2)
plt.plot(
    mae_kernel,
)
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.title("Blind Richardson-Lucy Deconvolution\nKernel MAE vs Iterations")
plt.show()

# %%
# Noisy grayscale with TV image prior example

img_size = 256
x = load_example(
    "SheppLogan.png",
    img_size=img_size,
    grayscale=True,
    device=device,
    resize_mode="resize",
)

# True physics (unknown kernel in blind setting)
# kernel_true = (
#     deepinv.utils.demo.load_degradation("Levin09.npy", index=7)
#     .unsqueeze(0)
#     .unsqueeze(0)
# )

kernel_true = deepinv.physics.functional.blur.gaussian_blur(
    psf_size=(25, 25), sigma=3.0
)
physics = deepinv.physics.BlurFFT(
    img_size=(1, img_size, img_size),
    noise_model=deepinv.physics.noise.PoissonNoise(gain=1 / 500, clip_positive=True),
    filter=kernel_true,
    device=device,
)

y = physics(x)

# Initial guesses
k0 = torch.ones_like(kernel_true)
x0 = y.clone()

max_iter = 100
x_hat, k_hat, xs, ks = blind_richardson_lucy(
    y=y,
    x0=x0,
    k0=k0,
    steps=max_iter,
    x_steps=1,
    k_steps=1,
    x_prior=deepinv.optim.TVPrior(),
    x_prior_weight=0.008,
    verbose=True,
    keep_inter=True,
    fft=False,
    normalize_kernel=True,
)
plot(
    [x, y, x_hat, kernel_true, k_hat],
    titles=[
        "Ground Truth",
        "Blurry image",
        "Blind Richardson-Lucy\nDeconvolution",
        "True kernel",
        "Estimated kernel",
    ],
    subtitles=[
        "PSNR (dB)",
        f"{psnr(x, y).item():.2f}",
        f"{psnr(x, x_hat).item():.2f}",
        "MAE",
        f"{mae(kernel_true, k_hat.cpu()).item():.4f}",
    ],
    figsize=(12, 4),
    rescale_mode="clip",
    vmin=vmin,
    vmax=vmax,
)

psnr_img = [psnr(x.cpu(), x_iter).item() for x_iter in xs]
mae_kernel = [mae(kernel_true.cpu(), k_iter).item() for k_iter in ks]

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(
    psnr_img,
)
plt.xlabel("Iteration")
plt.ylabel("PSNR (dB)")
plt.title("Blind Richardson-Lucy Deconvolution\nImage PSNR vs Iterations")
plt.subplot(1, 2, 2)
plt.plot(
    mae_kernel,
)
plt.xlabel("Iteration")
plt.ylabel("MAE")
plt.title("Blind Richardson-Lucy Deconvolution\nKernel MAE vs Iterations")
plt.show()

# %%
