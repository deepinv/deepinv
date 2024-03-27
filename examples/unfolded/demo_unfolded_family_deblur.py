# %%
r"""
Unfolded Douglas-Rachford for image deblurring
====================================================================================================

"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import deepinv as dinv
from deepinv.utils.demo import load_dataset
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import unfolded_builder
from deepinv.training_utils import train, test
from deepinv.physics import (
    MotionBlurGenerator,
    DiffractionBlurGenerator,
    GeneratorMixture,
    Blur,
    GaussianNoise,
)
from deepinv.models import DRUNet
from tqdm import tqdm
import torch.nn.functional as F
from deepinv.optim.utils import conjugate_gradient

# %%
# Setup paths for data loading and results.
# --------------------------------------------
#

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(42)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda:0")
dtype = torch.float32
factory_kwargs = {"device": device, "dtype": dtype}
# %%
# Load base image datasets and degradation operators.
# --------------------------------------------------------------------------------------------
# In this example, we use the CBSD68 dataset for training and the set3c dataset for testing.

operation = "deblurring"
train_dataset_name = "CBSD68"
test_dataset_name = "set3c"
img_size = 128 if torch.cuda.is_available() else 32
batch_size = 4
num_workers = 8

test_transform = transforms.Compose(
    [transforms.CenterCrop(img_size), transforms.ToTensor()]
)
train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)

train_dataset = load_dataset(
    train_dataset_name, ORIGINAL_DATA_DIR, transform=train_transform
)
test_dataset = load_dataset(
    test_dataset_name, ORIGINAL_DATA_DIR, transform=test_transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)

# %%
# Set up the reconstruction network
# --------------------------------------------------------
# Unrolled optimization algorithm parameters
# The parameters are initialized with a list of length max_iter, so that a distinct parameter is trained for each iteration.
max_iter = 4
stepsize = [1.0] * max_iter  # stepsize of the algorithm
sigma_denoiser = [0.03] * max_iter  # noise level parameter of the denoiser
beta = 1.0  # relaxation parameter of the Douglas-Rachford splitting
params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "beta": beta,
}
trainable_params = [
    "g_param",
    "stepsize",
]  # define which parameters from 'params_algo' are trainable

data_fidelity = L2()
prior = PnP(denoiser=DRUNet(train=True).to(device))

# Define the unfolded trainable model.


def tikhonov(y, physic):
    operator = lambda x: physic.A_adjoint(physic.A(x)) + 5e-2 * x
    b = physic.A_adjoint(y)
    x_init = conjugate_gradient(operator, b, max_iter=40)
    u_init = y
    return {"est": (x_init, u_init)}


model = unfolded_builder(
    iteration="DRS",
    params_algo=params_algo.copy(),
    trainable_params=trainable_params,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    custom_init=tikhonov,
    prior=prior,
    g_first=True,
)
print(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad is True)}"
)

# %% Generate blur operator
kernel_size = 15
kernel_init = torch.zeros((batch_size, 1, kernel_size, kernel_size), **factory_kwargs)
noise_model = GaussianNoise(0.03)
physic = Blur(
    filter=kernel_init,
    padding="valid",
    device=device,
    noise_model=noise_model,
    max_iter=40,
    tol=1e-5,
)
motion_gen = MotionBlurGenerator(
    shape=(batch_size, 3, kernel_size, kernel_size), device=device
)
diffraction_gen = DiffractionBlurGenerator(
    shape=(batch_size, 3, kernel_size, kernel_size), device=device
)
generator = GeneratorMixture([motion_gen, diffraction_gen], probs=c)
generator.step()

# dinv.utils.plot(physic.params)
x = next(iter(train_dataloader))[0].to(**factory_kwargs)
y = physic(x)
x_hat = model(y, physic)
dinv.utils.plot(x)
dinv.utils.plot(y)
dinv.utils.plot(x_hat)
with torch.no_grad():
    print(x_hat.min(), x_hat.max())
    est = tikhonov(y, physic)
    dinv.utils.plot(est["est"][0])
# %% Optimization parameters
# num_epochs = 1000
num_epochs = 10
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * len(train_dataloader), eta_min=1e-6
)


# %%
# Train the model
# ---------------
for epoch in range(num_epochs):
    progress_bar = tqdm(train_dataloader)
    avg_loss = 0.0
    for i, (x, _) in enumerate(progress_bar):
        optimizer.zero_grad()
        x = x.to(device)
        # Random generate a PSF for training
        kernel = generator.step()
        # Measuring the blurry images
        y = physic(x, theta=kernel)
        # Compute the estimation
        x_hat = model(y, physic)

        # Optimization
        loss = F.l1_loss(x_hat, x)
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        progress_bar.set_description(
            f"Epoch: {epoch + 1:02d} | Iteration: {i + 1:04d} | Avg Loss: {avg_loss / (i + 1):.4f}"
        )

    if (epoch + 1) % 10 == 0:
        # dinv.utils.plot(x)
        # dinv.utils.plot(y)
        dinv.utils.plot(x_hat)

# %%
import numpy as np

torch.backends.cudnn.benchmark = True

m = 10
step = 11
n_list = 2**7 * (4 + np.array([0, 1, 2, 3, 4]))
bs = 4

from torch.utils.benchmark import Timer

model.eval()
with torch.no_grad():
    for j, n in enumerate(n_list):
        x = torch.randn((bs, 3, n, n), device=device, dtype=dtype)
        y = physic(x)

        timer = Timer(stmt="model(y, physic)", globals=globals(), num_threads=8)

        t = timer.blocked_autorange(min_run_time=5)
        print("size: %i -- time: %1.3e " % (n, t.median / (n**2)))
