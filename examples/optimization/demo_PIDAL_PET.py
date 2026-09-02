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
)

x, attenuation = generate_pet_phantom(img_size, device=device)

background = None
physics.update(attenuation=attenuation, background=background)
y = physics(x)

prior = dinv.optim.prior.Tikhonov()

data_fidelity = dinv.optim.PoissonLikelihood(
    gain=gain
)

# def custom_init(y, physics):

#     x_init = x
#     z_init_1 = torch.clone(y)
#     z_init_2 = torch.clone(x_init)
#     z_init_3 = torch.clone(x_init)
#     u_init_1 = torch.clone(y)
#     u_init_2 = torch.clone(x_init)
#     u_init_3 = torch.clone(x_init)

#     return {"est": (x_init, (z_init_1, z_init_2, z_init_3), (u_init_1, u_init_2, u_init_3))}


pidal = dinv.optim.PIDAL(
    data_fidelity=data_fidelity,
    prior=prior,
    max_iter=10,
    stepsize=0.5,
    # custom_init=custom_init,
    # g_param=0.0
)

x_pidal = pidal.forward(
    y=y,
    physics=physics
)

dinv.utils.plot(
    [x, x_pidal],
    titles=["Ground truth", "PIDAL reconstruction"],
    figsize=(8, 4),
    rescale_mode="clip",
    vmin=0,vmax=1
)