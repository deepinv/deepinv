r"""
Saving and loading models
=========================

Models can be saved and loaded in the same way as in PyTorch. In this example, we show how to define, load and save a
model. For the purpose of the example, we choose an unfolded Chambolle Pock algorithm as the model.
The architecture of the model and its training are described
in the `constrained unfolded demo <https://deepinv.github.io/deepinv/auto_examples/unfolded/demo_unfolded_constrained_LISTA.html>`_.

"""

import importlib.util
from pathlib import Path
import torch

import deepinv as dinv
from deepinv.optim.data_fidelity import IndicatorL2
from deepinv.optim.prior import PnP
from deepinv.unfolded import unfolded_builder
from deepinv.models.utils import get_weights_url


# %%
# Setup paths for data loading and results
# ----------------------------------------
#

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"
CKPT_DIR = BASE_DIR / "ckpts"


# %%
# Define a forward operator
# -------------------------
# We define a simple inpainting operator with 50% of missing pixels.
#

n_channels = 3
img_size = 32
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Define the physics model
physics = dinv.physics.Inpainting(
    (n_channels, img_size, img_size), mask=0.5, device=device
)


# %%
# Define a model
# --------------
# For the purpose of this example, we define a rather complex model that consists an unfolded Chambolle-Pock algorithm.
#


# Select the data fidelity term
data_fidelity = IndicatorL2(radius=0.0)

# Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
# If the prior is initialized with a list of length max_iter,
# then a distinct weight is trained for each CP iteration.
# For fixed trained model prior across iterations, initialize with a single model.

level = 3
max_iter = 20  # Number of unrolled iterations

prior = [
    PnP(denoiser=dinv.models.WaveletDenoiser(wv="db8", level=level, device=device))
    for i in range(max_iter)
]

# Unrolled optimization algorithm parameters
lamb = [
    1.0
] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
stepsize = [
    1.0
] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.
sigma_denoiser = [0.01 * torch.ones(level, 3)] * max_iter

sigma = 1.0  # stepsize for Chambolle-Pock

params_algo = {
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
    "sigma": sigma,
    "K": physics.A,
    "K_adjoint": physics.A_adjoint,
}

trainable_params = [
    "g_param",
    "stepsize",
]  # define which parameters from 'params_algo' are trainable


# Because the CP algorithm uses more than 2 variables, we need to define a custom initialization.
def custom_init_CP(y, physics):
    x_init = physics.A_adjoint(y)
    u_init = y
    return {"est": (x_init, x_init, u_init)}


# Define the unfolded trainable model.
model = unfolded_builder(
    "CP",
    trainable_params=trainable_params,
    params_algo=params_algo,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior,
    g_first=False,
    custom_init=custom_init_CP,
)

# %%
# Saving the model
# ----------------
# We can save the trained model following the standard PyTorch procedure.

# Save the model

torch.save(model.state_dict(), CKPT_DIR / "inpainting/model_nontrained.pth")

# %%
# Loading the model
# -----------------
# Similarly, we can load our trained unfolded architecture following the standard PyTorch procedure.
# This network was trained in the demo :ref:`sphx_glr_auto_examples_unfolded_demo_unfolded_constrained_LISTA.py`.

# Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
# If the prior is initialized with a list of length max_iter,
# then a distinct weight is trained for each PGD iteration.
# For fixed trained model prior across iterations, initialize with a single model.

prior_new = [
    PnP(denoiser=dinv.models.WaveletDenoiser(wv="db8", level=level, device=device))
    for i in range(max_iter)
]

# Unrolled optimization algorithm parameters
lamb = [
    1.0
] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
stepsize = [
    1.0
] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.
sigma_denoiser = [0.01 * torch.ones(level, 3)] * max_iter

sigma = 1.0  # stepsize for Chambolle-Pock

params_algo_new = {
    "stepsize": stepsize,
    "g_param": sigma_denoiser,
    "lambda": lamb,
    "sigma": sigma,
    "K": physics.A,
    "K_adjoint": physics.A_adjoint,
}

model_new = unfolded_builder(
    "CP",
    trainable_params=trainable_params,
    params_algo=params_algo_new,
    data_fidelity=data_fidelity,
    max_iter=max_iter,
    prior=prior_new,
    g_first=False,
    custom_init=custom_init_CP,
)
print(
    "Parameter model_new.params_algo.g_param[0] at init: \n",
    model_new.params_algo.g_param[0],
)


# load a state_dict checkpoint
file_name = (
    "demo_unfolded_CP_ptwt.pth"
    if importlib.util.find_spec("ptwt")
    else "demo_unfolded_CP.pth"
)
url = get_weights_url(model_name="demo", file_name=file_name)
ckpt_state_dict = torch.hub.load_state_dict_from_url(
    url, map_location=lambda storage, loc: storage, file_name=file_name
)

# load a state_dict checkpoint
model_new.load_state_dict(ckpt_state_dict)

print(
    "Parameter model_new.params_algo.g_param[0] after loading: \n",
    model_new.params_algo.g_param[0],
)
