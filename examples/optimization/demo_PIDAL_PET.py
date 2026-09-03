# %%
import deepinv as dinv
from deepinv.utils.phantoms import generate_pet_phantom
import torch

img_size = (160, 160)
voxel_size = (2.0, 2.0)

device = "cuda" if torch.cuda.is_available() else "cpu"

gain = 0.01

physics = dinv.physics.PET(
    img_size=img_size,
    voxel_size=voxel_size,
    device=device,
    gain=gain,
    normalize=True,
    normalize_counts=True,
)

x, attenuation = generate_pet_phantom(img_size, device=device)

background = None
physics.update(attenuation=attenuation, background=background)
y = physics(x)

prior = dinv.optim.prior.TVPrior(n_it_max=200)

data_fidelity = dinv.optim.PoissonLikelihood(gain=gain)


pidal = dinv.optim.PIDAL(
    data_fidelity=data_fidelity,
    prior=prior,
    max_iter=100,
    stepsize=0.5,
    f_max_iter=10,
    lambda_reg=0.03
)

x_pidal = pidal.forward(y=y, physics=physics)

dinv.utils.plot(
    [x, x_pidal],
    titles=["Ground truth", "PIDAL reconstruction"],
    figsize=(8, 4),
)

# %%
