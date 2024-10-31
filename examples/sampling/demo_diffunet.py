r"""
Fixing diffunet
====================

Goal: fix diffunet implementation
"""

# %% Installing dependencies
# -----------------------------
# Let us ``import`` the relevant packages, and load a sample
# image of size 64x64. This will be used as our ground truth image.
# .. note::
#           We work with an image of size 64x64 to reduce the computational time of this example.
#           The algorithm works best with images of size 256x256.
#

import numpy as np
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url
from tqdm import tqdm  # to visualize progress

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = get_image_url("00000.png")
x_true = load_url_image(url=url, img_size=256).to(device)
x_true = x_true[:, :3, ...]  # remove alpha channel if present
x = x_true.clone()

# %%
# In this tutorial we consider random inpainting as the inverse problem, where the forward operator is implemented
# in :meth:`deepinv.physics.Inpainting`. In the example that we use, 90% of the pixels will be masked out randomly,
# and we will additionally have Additive White Gaussian Noise (AWGN) of standard deviation  12.75/255.

# first, the original model
# Now the original model

from deepinv.diffunet.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from deepinv.diffunet import utils_model

model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
args = utils_model.create_argparser(model_config).parse_args([])
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys()))

model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

x = (sqrt_alphas_cumprod[t_i] * y + sqrt_1m_alphas_cumprod[t_i] * torch.randn_like(x)) * mask \
                                + (1-mask) * x

x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
        model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)


model = dinv.models.DiffUNet(large_model=False).to(device)

num_train_timesteps = 1000  # Number of timesteps used during training


def get_betas(
    beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=num_train_timesteps
):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas = torch.from_numpy(betas).to(device)

    return betas


# Utility function to let us easily retrieve \bar\alpha_t
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


betas = get_betas()

time_step_int = 200

t = torch.ones(1, device=device) * time_step_int  # choose some arbitrary timestep
at = compute_alpha(betas, t.long())
sigmat = (1 - at).sqrt() / at.sqrt()

x0 = x_true
xt = x0 + sigmat * torch.randn_like(x0)

# apply denoiser
x0_t = model(xt, sigmat)

# Visualize
imgs = [x0, xt, x0_t]
plot(
    imgs,
    titles=["ground-truth", "noisy", "posterior mean"],
)


