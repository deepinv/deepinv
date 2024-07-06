import sys

sys.path.append("/home/zhhu/workspaces/deepinv/")

from datetime import datetime
import deepinv as dinv
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot, plot_curves
from deepinv.optim.phase_retrieval import (
    correct_global_phase,
    cosine_similarity,
    spectral_methods,
    default_preprocessing,
)
from deepinv.models.complex import to_complex_denoiser

now = datetime.now()
dt_string = now.strftime("%Y%m%d-%H%M%S")

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = DATA_DIR / dt_string
FIGURE_DIR = DATA_DIR / "first_results"
LOAD_DIR = DATA_DIR / "latest" / "pseudorandom"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR / "random").mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR / "pseudorandom").mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
print("device is: ", device)
print(SAVE_DIR)

# Set up the variable to fetch dataset and operators.
img_size = 99
url = get_image_url("SheppLogan.png")
x = load_url_image(
    url=url, img_size=img_size, grayscale=True, resize_mode="resize", device=device
)
x.shape

# generate phase signal

# The phase is computed as 2*pi*x - pi, where x is the original image.
x_phase = torch.exp(1j * x * torch.pi - 0.5j * torch.pi).to(device)

# Every element of the signal should have unit norm.
assert torch.allclose(x_phase.real**2 + x_phase.imag**2, torch.tensor(1.0))

repeat = 100

# 1-99, 99-141, 141-299
start = 141
end = 299

res_gd_spec = torch.empty((end - start) // 2, repeat)
oversampling_ratios = torch.empty((end - start) // 2)

data_fidelity = L2()
prior = dinv.optim.prior.Zero()
max_iter = 10000
early_stop = True
verbose = True
# stepsize: use 1e-4 for oversampling ratio 0-2, and 3e-3*oversampling for oversampling ratio 2-9
step_size = 3e-3

n_layers = 2


def spectral_methods_wrapper(y, physics, **kwargs):
    x = spectral_methods(y, physics, n_iter=5000, **kwargs)
    z = spectral_methods(y, physics, n_iter=5000, **kwargs)
    return {"est": (x, z)}


for i in trange(start, end, 2):
    params_algo = {"stepsize": step_size * i**2 / (99**2), "g_params": 0.00}
    print("stepsize:", params_algo["stepsize"])
    model = optim_builder(
        iteration="PGD",
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=max_iter,
        verbose=verbose,
        params_algo=params_algo,
        custom_init=spectral_methods_wrapper,
    )
    for j in range(repeat):
        physics = dinv.physics.PseudoRandomPhaseRetrieval(
            n_layers=n_layers,
            input_shape=(1, img_size, img_size),
            output_shape=(1, i, i),
            dtype=torch.cfloat,
            device=device,
        )
        y = physics(x_phase)

        oversampling_ratios[(i - start) // 2] = physics.oversampling_ratio

        x_phase_gd_spec, _ = model(y, physics, x_gt=x_phase, compute_metrics=True)

        res_gd_spec[(i - start) // 2, j] = cosine_similarity(x_phase, x_phase_gd_spec)
        print(res_gd_spec[(i - start) // 2, j])

# save results
torch.save(
    res_gd_spec, SAVE_DIR / "pseudorandom" / "res_gd_spec_2-9_2layer_100repeat.pt"
)
torch.save(
    oversampling_ratios,
    SAVE_DIR / "pseudorandom" / "oversampling_ratios_gd_spec_2-9_2layer_100repeat.pt",
)
