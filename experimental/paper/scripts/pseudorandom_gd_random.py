import sys

sys.path.append("/home/zhhu/workspaces/deepinv/")

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm, trange

import deepinv as dinv
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

# genral
model_name = "pseudorandom"
recon = "gd-rand"

# pseudorandom settings
img_size = 64
n_layers = 1
shared_weights = False
drop_tail = True

# optim settings
data_fidelity = L2()
prior = dinv.optim.prior.Zero()
early_stop = True
verbose = True
n_repeats = 100
n_iter = 10000
# stepsize: use 1e-4 for oversampling ratio 0-2, and 3e-3*oversampling for oversampling ratio 2-9
step_size = 1e-8
start = 90
end = 144
output_sizes = torch.arange(start, end, 2)
oversampling_ratios = output_sizes**2 / img_size**2
n_oversampling = oversampling_ratios.shape[0]

# save settings
res_name = f"res_{model_name}_{n_layers}_{recon}_{n_repeats}repeat_{n_iter}iter_{oversampling_ratios[0].numpy()}-{oversampling_ratios[-1].numpy()}.csv"

print("res_name:", res_name)

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "runs"
SAVE_DIR = DATA_DIR / current_time
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR / "random").mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR / "pseudorandom").mkdir(parents=True, exist_ok=True)
print("save directory:", SAVE_DIR)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
device

# Set up the variable to fetch dataset and operators.
url = get_image_url("SheppLogan.png")
x = load_url_image(
    url=url, img_size=img_size, grayscale=True, resize_mode="resize", device=device
)

# generate phase signal
# The phase is computed as 2*pi*x - pi, where x is the original image.
x_phase = torch.exp(1j * x * torch.pi - 0.5j * torch.pi).to(device)
# Every element of the signal should have unit norm.
assert torch.allclose(x_phase.real**2 + x_phase.imag**2, torch.tensor(1.0))

df_res = pd.DataFrame(
    {
        "oversampling_ratio": oversampling_ratios,
        "step_size": None,
        **{f"repeat{i}": None for i in range(n_repeats)},
    }
)


def random_init(y, physics):
    x = torch.randn_like(x_phase)
    z = x.detach().clone()
    return {"est": (x, z)}


for i in trange(n_oversampling):
    oversampling_ratio = oversampling_ratios[i]
    output_size = output_sizes[i]
    print(f"output_size: {output_size}")
    print(f"oversampling_ratio: {oversampling_ratio}")
    step_size = step_size * oversampling_ratio
    df_res.loc[i, "step_size"] = step_size
    params_algo = {"stepsize": step_size.item(), "g_params": 0.00}
    print("stepsize:", params_algo["stepsize"])
    model = optim_builder(
        iteration="PGD",
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=n_iter,
        verbose=verbose,
        params_algo=params_algo,
        custom_init=random_init,
    )
    for j in range(n_repeats):
        physics = dinv.physics.PseudoRandomPhaseRetrieval(
            n_layers=n_layers,
            input_shape=(1, img_size, img_size),
            output_shape=(1, output_size, output_size),
            dtype=torch.cfloat,
            device=device,
            shared_weights=shared_weights,
            drop_tail=drop_tail,
        )
        y = physics(x_phase)

        x_phase_gd_spec = model(y, physics, x_gt=x_phase)
        df_res.loc[i, f"repeat{j}"] = cosine_similarity(x_phase, x_phase_gd_spec).item()
        # print the cosine similarity
        print(f"cosine similarity: {df_res.loc[i, f'repeat{j}']}")

# save results
df_res.to_csv(SAVE_DIR / model_name / res_name)
