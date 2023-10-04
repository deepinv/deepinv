import deepinv as dinv
from deepinv.utils import load_url_image

device='cpu'
url = "https://www-iuem.univ-brest.fr/intranet/communication/logos/tutelles-iuem/cnrs/cnrs-poster.png"
x = load_url_image(url=url, img_size=64, grayscale=True, device=device)

physics = dinv.physics.BlurFFT(
    img_size=(3, 64, 64),
    filter=dinv.physics.blur.gaussian_blur(5,5),
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.03),
)

data_fidelity = dinv.optim.data_fidelity.L2()
prior = dinv.optim.prior.PnP(denoiser=dinv.models.MedianFilter())
model = dinv.optim.optim_builder(iteration="PGD",
                                prior=prior, 
                                data_fidelity=data_fidelity,
                                params_algo={"stepsize": 1.0, "g_param": 0.1, "lambda": 2.},
                                g_first=True,
                                )

y = physics(x)
x_hat = model(y, physics)
dinv.utils.plot([x, y, x_hat], ["signal", "measurement", "estimate"], rescale_mode='clip')