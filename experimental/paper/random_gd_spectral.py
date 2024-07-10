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
    spectral_methods_wrapper,
)
from deepinv.models.complex import to_complex_denoiser

repeat = 30
max_iter = 5000
step_size = 5e-3
#oversampling_ratios = torch.cat((torch.arange(0.1,4.1,0.1),torch.arange(4.2,9.2,0.4)))
#oversampling_ratios = torch.arange(0.1, 2.1, 0.1)
oversampling_ratios = torch.cat((torch.arange(2.1, 5.1, 0.1),torch.arange(5.2, 9.2, 0.2)))
print("oversampling_ratios:", oversampling_ratios)

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

res_gd_spec = torch.empty(oversampling_ratios.shape[0], repeat)

data_fidelity = L2()
prior = dinv.optim.prior.Zero()
early_stop = True
verbose = True


for i in trange(oversampling_ratios.shape[0]):
    oversampling_ratio = oversampling_ratios[i]
    print(f"oversampling_ratio: {oversampling_ratio}")
    params_algo = {"stepsize": (step_size * oversampling_ratio).item(), "g_params": 0.00}
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
        physics = dinv.physics.RandomPhaseRetrieval(
            m=int(oversampling_ratio * torch.prod(torch.tensor(x_phase.shape))),
            img_shape=(1, img_size, img_size),
            dtype=torch.cfloat,
            device=device,
        )
        y = physics(x_phase)

        x_phase_gd_spec, _ = model(y, physics, x_gt=x_phase, compute_metrics=True)

        res_gd_spec[i, j] = cosine_similarity(x_phase, x_phase_gd_spec)
        print(res_gd_spec[i, j])

# save results
torch.save(
    res_gd_spec, SAVE_DIR / "random" / "res_gd_spec_2-9_30repeat.pt"
)
torch.save(
    oversampling_ratios,
    SAVE_DIR / "random" / "oversampling_ratios_gd_spec_2-9_30repeat.pt",
)