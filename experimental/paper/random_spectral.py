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
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR / "random").mkdir(parents=True, exist_ok=True)
Path(SAVE_DIR / "pseudorandom").mkdir(parents=True, exist_ok=True)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
device

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

start = 1
end = 91

res_spec = torch.empty((end - start) // 2, repeat)
oversampling_ratios = torch.empty((end - start) // 2)

for i in trange(start, end):
    oversampling_ratio = i / 10
    print(f"oversampling_ratio: {oversampling_ratio}")
    for j in range(repeat):
        physics = dinv.physics.RandomPhaseRetrieval(
            m=int(oversampling_ratio * img_size ** 2),
            img_shape=(1, img_size, img_size),
            dtype=torch.cfloat,
            device=device,
        )
        y = physics(x_phase)

        oversampling_ratios[i-1] = oversampling_ratio
        x_phase_spec = spectral_methods(y, physics, n_iter=5000)
        res_spec[i-1, j] = cosine_similarity(x_phase, x_phase_spec)
        # print the cosine similarity
        print(f"cosine similarity: {res_spec[i-1,j]}")

# save results
torch.save(res_spec, SAVE_DIR / "random" / "res_spec_100repeat.pt")
torch.save(
    oversampling_ratios,
    SAVE_DIR / "random" / "oversampling_ratios_spec_100repeat.pt",
)