
import torch
import deepinv as dinv
from deepinv.models import NCSNpp

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64
figsize = 2.5
gif_frequency = 10  # Increase this value to save the GIF saving time
# %%
from deepinv.sampling import (
    PosteriorDiffusion,
    DPSDataFidelity,
    EulerSolver,
    VarianceExplodingDiffusion,
)
from deepinv.optim import ZeroFidelity

# In this example, we use the pre-trained FFHQ-64 model from the
# EDM framework: https://arxiv.org/pdf/2206.00364 .
# The network architecture is from Song et al: https://arxiv.org/abs/2011.13456 .
denoiser = NCSNpp(pretrained="download").to(device)

# The solution is obtained by calling the SDE object with a desired solver (here, Euler).
# The reproducibility of the SDE Solver class can be controlled by providing the pseudo-random number generator.
num_steps = 150
rng = torch.Generator(device).manual_seed(42)
timesteps = torch.linspace(1, 0.001, num_steps)
solver = EulerSolver(timesteps=timesteps, rng=rng)


sigma_min = 0.005
sigma_max = 5
sde = VarianceExplodingDiffusion(
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    alpha=0.5,
    device=device,
    dtype=dtype,
)

mask = torch.ones_like(x)
mask[..., 24:40, 24:40] = 0.0
physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=mask, device=device)
y = physics(x)

weight = 1.0  # guidance strength
dps_fidelity = DPSDataFidelity(denoiser=denoiser, weight=weight)

model = PosteriorDiffusion(
    data_fidelity=dps_fidelity,
    denoiser=denoiser,
    sde=sde,
    solver=solver,
    dtype=dtype,
    device=device,
    verbose=True,
)

# To perform posterior sampling, we need to provide the measurements, the physics and the solver.
# Moreover, when the physics is given, the initial point can be inferred from the physics if not given explicitly.
seed_1 = 11
x_hat, trajectory = model(
    y,
    physics,
    seed=seed_1,
    get_trajectory=True,
)
# Here, we plot the original image, the measurement and the posterior sample
dinv.utils.plot(
    [x, y, x_hat],
    show=True,
    titles=["Original", "Measurement", "Posterior sample"],
    save_fn="posterior_sample.png",
    figsize=(figsize * 3, figsize),
)
# We can also save the trajectory of the posterior sample
dinv.utils.save_videos(
    trajectory[::gif_frequency],
    time_dim=0,
    titles=["Posterior sample with VE"],
    save_fn="posterior_trajectory.gif",
    figsize=(figsize, figsize),
)

