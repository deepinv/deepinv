# %%
import deepinv as dinv

device = "cuda"

prior = dinv.optim.prior.Tikhonov()
# prior = dinv.optim.prior.L1Prior()

rl = dinv.optim.MLEM(
    data_fidelity=None,
    prior=None,
    lambda_reg=0.5,
    max_iter=100,
    crit_conv=None,
    thres_conv=1e-5,
    early_stop=False,
    backtracking=False,
    custom_metrics=None,
    custom_init=None,
    unfold=False,
    trainable_params=None,
    DEQ=False,
)

x = dinv.utils.load_example("butterfly.png", img_size=(256, 256), device=device)
psf = dinv.physics.blur.gaussian_blur(sigma=2)
noise_model = dinv.physics.noise.GaussianNoise(sigma=15 / 255)

physics = dinv.physics.Blur(
    filter=psf, padding="circular", noise_model=noise_model, device=device
)

y = physics(x)
x_rl = rl(y, physics, compute_metrics=False)

dinv.utils.plot([x, y, x_rl], rescale_mode="clip", vmin=0, vmax=1)


# %%
