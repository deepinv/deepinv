r"""
Weakly Convex Ridge Regularizer
===================================================

...

"""

from deepinv.models import RidgeRegularizer
import torch
from deepinv.utils.demo import load_url_image, get_image_url
import numpy as np
from deepinv.utils.plotting import plot
from deepinv.physics import Inpainting

device = "cuda"

url = get_image_url("CBSD_0010.png")
x = load_url_image(url, grayscale=True).to(device)
physics = Inpainting(.5).to(device)
noise_level = 0.01

y=physics(x)

noisy = y + noise_level * torch.randn_like(y)

model = RidgeRegularizer().to(device)

with torch.no_grad():
    recon=model.reconstruct(physics,y,0.05,1.)
plot([x,y,recon], titles=["ground truth","observation","reconstruction"])
exit()

"""
multiconv_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/multiconv.pt',map_location='cpu')
model.W.load_state_dict(multiconv_dict)
alpha_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/alpha.pt',map_location='cpu')
model.potential.alpha_spline.load_state_dict(alpha_dict)
mu_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/mu.pt',map_location='cpu')
model.potential.mu_spline.load_state_dict(mu_dict)
phi_plus_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/phi_plus.pt',map_location='cpu')
model.potential.phi_plus.load_state_dict(phi_plus_dict)
phi_minus_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/phi_minus.pt',map_location='cpu')
model.potential.phi_minus.load_state_dict(phi_minus_dict)

all_weights=model.state_dict()
torch.save(all_weights,'../../deepinv/saved_model/weights.pt')
exit()
"""

model.load_state_dict(torch.load("../../deepinv/saved_model/weights.pt"))
mu_dict = torch.load(
    "../../deepinv/saved_model/saved_model_WCRR/mu.pt", map_location="cpu"
)
# model.potential.mu_spline.load_state_dict(mu_dict)
# print(model.potential.mu_spline.coefficients)
# exit()

grad = model.grad(noisy, noise_level)

with torch.no_grad():
    recon = model(noisy, noise_level)
plot([recon], titles=["reconstruction"])
exit()
with torch.no_grad():
    recon = torch.clone(noisy)
    for step in range(4000):
        grad_reg = model.grad(recon, noise_level)
        grad_data = recon - noisy
        full_grad = grad_data + grad_reg
        recon = recon - 1e-3 * full_grad
        # recon=torch.maximum(recon,torch.zeros(1).to(device))
        if step % 10 == 0:
            print(model.cost(recon, noise_level))
plot([recon], titles=["reconstruction"])
exit()


def accelerated_gd_single(
    x_noisy,
    model,
    sigma,
    ada_restart=False,
    stop_condition=None,
    lmbd=1,
    grad_op=None,
    t_init=1,
    **kwargs,
):

    max_iter = kwargs.get("max_iter", 500)
    tol = kwargs.get("tol", 1e-4)

    # initial value: noisy image
    x = torch.clone(x_noisy)
    z = torch.clone(x_noisy)
    t = t_init
    # the index of the images that have not converged yet
    # relative change in the estimate
    res = 100000

    mu = torch.exp(
        model.potential.mu_spline(torch.tensor([[[[sigma * 255]]]], device=device))
    )
    # mu,scaling=model.potential.get_mu_scaling(torch.tensor([sigma*255],device=device))

    # print(mu)
    # print(scaling)
    # exit()
    step_size = 1 / (1 + lmbd * mu)
    for i in range(max_iter):

        x_old = torch.clone(x)

        grad = lmbd * model.grad(z, sigma) + (z - x_noisy)
        grad = grad * step_size

        x = z - grad

        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        z = x + (t_old - 1) / t * (x - x_old)

        if i > 0:
            res = (torch.norm(x - x_old) / (torch.norm(x))).item()

    return (x, i, t)


with torch.no_grad():
    out = accelerated_gd_single(noisy, model, noise_level)[0]

plot([x, noisy, out], titles=["ground truth", "noisy", "recon"])
