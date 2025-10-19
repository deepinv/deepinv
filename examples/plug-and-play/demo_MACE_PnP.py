"""
Multi-Agent Consensus Equilibrium (MACE) for Image Deblurring
===============================================================

This example demonstrates solving an image deblurring problem using the
Multi-Agent Consensus Equilibrium (MACE) framework within DeepInv.

MACE is a powerful algorithm that generalizes methods like Douglas-Rachford
Splitting to handle multiple "agents" (data fidelity terms or priors)
concurrently. At each iteration, MACE applies all agent operators in parallel
(the F-step) and then computes a weighted average of their outputs to reach
a consensus (the G-step). A Mann iteration scheme with relaxation parameter
``rho`` is used for convergence.

This specific example showcases MACE with three agents:
1. An L2 data fidelity term for the blur operator.
2. A Plug-and-Play (PnP) prior using a pre-trained DnCNN denoiser.
3. A Total Variation (TV) prior for regularization.

The relative influence of each agent is controlled by the ``mu`` weights.
"""

from pathlib import Path
import torch
import deepinv as dinv
from deepinv.models import DnCNN, DRUNet

from deepinv.utils.demo import load_example
from deepinv.utils.plotting import plot, plot_curves
from deepinv.optim.prior import PnP, TVPrior
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

IMG_SIZE = 256
IMG_NAME = "barbara.jpeg"
CHANNELS = 3  # 1 for grayscale, 3 for color
DEVICE = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
NOISE_STD = 0.1
BLUR_SIGMA = (2.0, 2.0)
BLUR_ANGLE = 0.0

# MACE Algorithm parameters
MAX_ITER = 30
RHO_MACE = 0.8 # Relaxation parameter for MACE (0 < rho <= 1)

# Weights for agents: [L2_DataFidelity, PnP_DnCNN_Prior, TV_Prior]
MU_MACE = [0.5, 0.49, 0.01] # Ensure sum to 1
STEPSIZE_MACE = 1.0
LAMBDA_PNP = 1
LAMBDA_TV = 1
G_PARAM_PNP = NOISE_STD 

# -----------------------------------------------------------------------------
# Load Image and Define Physics
# -----------------------------------------------------------------------------

torch.manual_seed(0)

x_gt = load_example(
    IMG_NAME,
    img_size=IMG_SIZE,
    grayscale=(CHANNELS == 1),
    resize_mode="resize",
    device=DEVICE,
)

filter_kernel = dinv.physics.blur.gaussian_blur(sigma=BLUR_SIGMA, angle=BLUR_ANGLE)
physics = dinv.physics.BlurFFT(
    img_size=(CHANNELS, IMG_SIZE, IMG_SIZE),
    filter=filter_kernel,
    device=DEVICE,
)

y_clean = physics(x_gt)
y_noisy = y_clean + NOISE_STD * torch.randn_like(y_clean)

# -----------------------------------------------------------------------------
# Define MACE Agents
# -----------------------------------------------------------------------------

# 1. Data Fidelity Agent (L2 Norm)
data_fidelity_agent = L2()

# 2. PnP Prior Agent (DnCNN Denoiser)
dncnn_denoiser = DnCNN(
    in_channels=CHANNELS,
    out_channels=CHANNELS,
    pretrained="download",
    device=DEVICE,
)
pnp_agent = PnP(denoiser=dncnn_denoiser)

# 3. TV Prior Agent
tv_agent = TVPrior()

drunet_denoiser = DRUNet(
    in_channels=CHANNELS,
    out_channels=CHANNELS,
    pretrained="download",
    device=DEVICE,
)

pnp_agent_drunet = PnP(denoiser=drunet_denoiser)

# Combine priors into a list for MACE
prior_agents_list = [pnp_agent, tv_agent]


# -----------------------------------------------------------------------------
# Configure and Build MACE Optimizer
# -----------------------------------------------------------------------------

params_algo_mace = {
    "stepsize": STEPSIZE_MACE,
    "g_param": G_PARAM_PNP,
    "rho": RHO_MACE,
    "mu": [MU_MACE],
    "lambda_0": LAMBDA_PNP,
    "lambda_1": LAMBDA_TV
}


mace_reconstructor = optim_builder(
    iteration="MACE",
    data_fidelity=data_fidelity_agent,
    prior=prior_agents_list,
    max_iter=MAX_ITER,
    early_stop=False,
    thres_conv=1e-5,
    verbose=True,
    params_algo=params_algo_mace,
    get_output=lambda X: X["est"][0],
    custom_init=lambda y, p: {"est": (p.A_adjoint(y),)}
)

# -----------------------------------------------------------------------------
# Run Reconstruction and Evaluate
# -----------------------------------------------------------------------------

print("Running MACE reconstruction...")
x_mace, metrics_mace = mace_reconstructor(y_noisy, physics, x_gt=x_gt, compute_metrics=True)

x_linear = physics.A_adjoint(y_noisy) # For comparison

# -----------------------------------------------------------------------------
# Results and Visualization
# -----------------------------------------------------------------------------

psnr_metric = dinv.metric.PSNR()
print(f"Linear reconstruction PSNR: {psnr_metric(x_linear, x_gt).item():.2f} dB")
print(f"MACE reconstruction PSNR  : {psnr_metric(x_mace, x_gt).item():.2f} dB")

RESULTS_DIR = Path("./results/MACE_PnP_Multi_Agent")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plot(
    [x_gt, y_noisy, x_linear, x_mace],
    titles=["Ground Truth", "Blurred and Noisy", "Linear Adjoint", "MACE (L2 + PnP + TV)"],
    save_dir=RESULTS_DIR / "images",
    show=True,
)


plot_curves(metrics_mace, save_dir=RESULTS_DIR / "curves", show=True)


