r"""
Inverse scattering problem
==========================

In this example we show how to use the :class:`deepinv.physics.Scattering` forward model.

"""

import deepinv as dinv
import torch

img_width = 64
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
x = dinv.utils.load_example("butterfly.png", img_size=img_width, device=device, grayscale=True)

contrast = 0.01
x = x * contrast

wavenumber = img_width / 4 * (2 * torch.pi)
sensors = 32

receivers = dinv.physics.scattering.circular_sensors(sensors, radius=1, device=device)
transmitters = dinv.physics.scattering.circular_sensors(sensors, radius=1, device=device)

physics = dinv.physics.Scattering(img_width=img_width, device=device, wavenumber=wavenumber,
                                  transmitters=transmitters, receivers=receivers)

y = physics(x)

x_hat = physics.A_dagger(y)

physics.normalize(x)
print("Operator norm:", physics.compute_norm(x))

denoiser = dinv.models.DRUNet(in_channels=1, out_channels=1, device=device)
denoiser = dinv.models.complex.to_complex_denoiser(denoiser, mode='real_imag')
prior = dinv.optim.PnP(denoiser=denoiser)
fid = dinv.optim.L2()
pnp_solver = dinv.optim.PGD(prior=prior, data_fidelity=fid, stepsize=0.5, sigma_denoiser=0.02, max_iter=20)

x_pnp = pnp_solver(y, physics)

dinv.utils.plot([x, x_hat, x_pnp], titles=['Original', 'Reconstruction', 'PnP Reconstruction'])