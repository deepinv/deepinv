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

model_name = "random-haar"
recon = "gd_spectral"
n_repeats = 100
n_iter = 10000
oversampling_ratios = torch.arange(2.0, 5.2, 0.2)
#oversampling_ratios = torch.cat((torch.arange(0.1,3.1,0.1),torch.arange(3.5,9.5,0.5)))
# oversampling_ratios = torch.cat((torch.arange(2.1, 5.1, 0.1),torch.arange(5.2, 9.2, 0.2)))
n_oversampling = oversampling_ratios.shape[0]
res_name = f"res_{model_name}_{recon}_{n_repeats}repeat_{n_iter}iter_{oversampling_ratios[0].numpy()}-{oversampling_ratios[-1].numpy()}.csv"
use_haar = True

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = Path("..")
DATA_DIR = BASE_DIR / "runs"
SAVE_DIR = DATA_DIR / current_time
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
print("save directory:", SAVE_DIR)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
device


# Set up the variable to fetch dataset and operators.
img_size = 64
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

data_fidelity = L2()
prior = dinv.optim.prior.Zero()
early_stop = True
verbose = True

for i in trange(oversampling_ratios.shape[0]):
    oversampling_ratio = oversampling_ratios[i]
    # if oversampling_ratio < 2.5:
    #     step_size = 1e-4
    # elif oversampling_ratio < 3.0:
    #     step_size = 3e-3
    # else:
    #     step_size = 1e-2
    if oversampling_ratio < 2.0:
        step_size = 1e-4
    else:
        step_size = 3e-3
    print(f"oversampling_ratio: {oversampling_ratio}")
    df_res.loc[i, "step_size"] = step_size
    params_algo = {
        "stepsize": (step_size * oversampling_ratio).item(),
        "g_params": 0.00,
    }
    print("stepsize:", params_algo["stepsize"])
    model = optim_builder(
        iteration="PGD",
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=n_iter,
        verbose=verbose,
        params_algo=params_algo,
        custom_init=spectral_methods_wrapper,
    )
    for j in range(n_repeats):
        physics = dinv.physics.RandomPhaseRetrieval(
            m=int(oversampling_ratio * torch.prod(torch.tensor(x_phase.shape))),
            img_shape=(1, img_size, img_size),
            dtype=torch.cfloat,
            device=device,
            use_haar=use_haar,
        )
        y = physics(x_phase)

        x_phase_gd_spec, _ = model(y, physics, x_gt=x_phase, compute_metrics=True)

        df_res.loc[i, f"repeat{j}"] = cosine_similarity(x_phase, x_phase_gd_spec).item()
        print(df_res.loc[i, f"repeat{j}"])

# save results
df_res.to_csv(SAVE_DIR / res_name)
