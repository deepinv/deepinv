# %%
import sys
sys.path.append("/home/ubuntu/workspaces/deepinv/")

# %%
import deepinv as dinv
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.optim.phase_retrieval import correct_global_phase, cosine_similarity
from deepinv.models.complex import to_complex_denoiser

# %%
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
device

x = torch.rand(1, 1, 50, 50, device=device)
# %%
# We use the original image as the phase information for the complex signal.
# The phase is computed as 2*pi*x - pi, where x is the original image.
x_phase = torch.exp(2j*x*torch.pi - 1j*torch.pi)

# Every element of the signal should have unit norm.
assert torch.allclose(x_phase.real**2 + x_phase.imag**2, torch.tensor(1.0))

# %%
def run_gd_rand(
        x,
        y,
        physics,
        oversampling,
        num_iter = 1500,
        stepsize = 0.04,
        data_fidelity = L2()
):
    stepsize = stepsize * oversampling
    x_gd_rand = torch.randn_like(x)

    for _ in range(num_iter):
        x_gd_rand = x_gd_rand - stepsize * data_fidelity.grad(x_gd_rand, y, physics)

    return x_gd_rand

def run_spec(
        y,
        physics,
        num_iter = 300,
):
    x_spec = physics.A_dagger(y, n_iter=num_iter)
    x_spec = x_spec * torch.sqrt(y.sum())

    return x_spec

def run_gd_spec(
        y,
        physics,
        oversampling,
        spec_iter = 300,
        num_iter = 1500,
        stepsize = 0.005,
        data_fidelity = L2() 
):
    stepsize = stepsize * oversampling

    x_gd_spec = run_spec(y, physics, spec_iter)

    for _ in range(num_iter):
        x_gd_spec = x_gd_spec - stepsize * data_fidelity.grad(x_gd_spec, y, physics)
    
    return x_gd_spec

def run_random_case(
        x,
        img_shape,
        noise_level=0.05,
        start=0.1,
        end=20.0,
        step=0.1,
        repeat=10,
        run=["gd_rand", "spec", "gd_spec"]
):
    x = x.to(device)
    
    oversampling_series = list(np.arange(start,end+step,step))
    count = len(oversampling_series)
    
    res_gd_rand = np.empty((count, repeat))
    res_spec = np.empty((count, repeat))
    res_gd_spec = np.empty((count, repeat))

    for i in tqdm(range(count)):
        oversampling = oversampling_series[i]
        m = int(oversampling * torch.prod(torch.tensor(img_shape)))
        for j in range(repeat):
            physics = dinv.physics.RandomPhaseRetrieval(
                m=m,
                img_shape=img_shape,
                noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
                device=device,
            )
            y = physics(x)
            
            if "gd_rand" in run:
                x_gd_rand = run_gd_rand(x, y, physics, oversampling)
                res_gd_rand[i,j] = cosine_similarity(x_gd_rand, x)
            if "spec" in run:
                x_spec = run_spec(y, physics)
                res_spec[i,j] = cosine_similarity(x_spec, x)
            if "gd_spec" in run:
                x_gd_spec = run_gd_spec(y, physics, oversampling)
                res_gd_spec[i,j] = cosine_similarity(x_gd_spec, x)
    
    # save results
    np.save("res_gd_rand.npy", res_gd_rand)
    np.save("res_spec.npy", res_spec)
    np.save("res_gd_spec.npy", res_gd_spec)

    return res_gd_rand, res_spec, res_gd_spec

# %%
start = 0.1
end = 2.0
step = 0.1

# run random case
res_gd_rand, res_spec, res_gd_spec = run_random_case(
    x=x_phase,
    img_shape=x.shape[1:],
    noise_level=0.05,
    start=start,
    end=end,
    step=step,
    repeat=10,
    run = ["spec"]
)

print(res_spec.shape)
# save results
np.save("res_spec_small.npy", res_spec)
# %%
plt.figure(figsize=(10,5))
#plt.plot(np.arange(1.0,2.1,0.1),np.mean(res_gd_rand, axis=1), label="GD Random")
plt.plot(np.arange(start,end+step,step),np.mean(res_spec, axis=1), label="Spec")
#plt.plot(np.arange(1.0,2.1,0.1),np.mean(res_gd_spec, axis=1), label="GD Spec")
plt.xlabel("Oversampling")
plt.ylabel("Cosine Similarity")
plt.legend()
plt.show()

