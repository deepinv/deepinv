import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import RED, PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.training_utils import test
from torchvision import transforms
from deepinv.utils.parameters import get_GSPnP_params, get_DPIR_params
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.optim.optim_iterators import PGDIteration
import matplotlib as mpl
from deepinv.utils import load_url_image
mpl.rcParams.update(mpl.rcParamsDefault)

BASE_DIR = Path(".")
ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
DATA_DIR = BASE_DIR / "measurements"
RESULTS_DIR = BASE_DIR / "results"
DEG_DIR = BASE_DIR / "degradations"

torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fdatasets&files=dbc66cf_62a2936fc8fd4b2aa6df2bc7a968888c-0-b662065fadbb4f35a5de7f0cd36adfbd.jpg"
)
x = load_url_image(url=url).to(device)

# Generate the degradation operator.
kernel_index = 1
kernel_torch = load_degradation(
    "kernels_12.npy", DEG_DIR / "kernels", kernel_index=kernel_index
)
kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
    0
)  # add batch and channel dimensions
operator = dinv.physics.BlurFFT(x.shape, filter=kernel_torch , device=device) # try changing the operator!
y = operator(x)

operation = "deblur"
noise_level_img = 0.03
data_fidelity = L2()


# load specific parameters for DPIR
lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_level_img)
params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
early_stop = False  # Do not stop algorithm with convergence criteria

# Select the data fidelity term
data_fidelity = L2()

# Specify the denoising prior
prior = PnP(denoiser=dinv.models.DRUNet(pretrained="download", train=False, device=device))


# instantiate the algorithm class to solve the IP problem.
model = optim_builder(
    iteration="HQS",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=True,
    params_algo=params_algo,
)

from deepinv.utils import cal_psnr
#x1, metrics, images = model(y, operator, x_gt=x, compute_metrics=True, get_all_iterations=True)
x1, metrics = model(y, operator, x_gt=x, compute_metrics=True, get_all_iterations=False)
x_init = operator.A_adjoint(y)
cur_psnr_init = cal_psnr(x_init, x)
cur_psnr = cal_psnr(x1, x)
print(cur_psnr_init,cur_psnr)

from deepinv.utils.plotting import plot_curves, plot, plot_animation
plot_curves(metrics)
# plot_animation(images = images, save_dir='images')
# plot_animation(metrics = metrics, metric_name='psnr', save_dir='curves')
# plot_animation(metrics = metrics, metric_name='residual', save_dir='curves')
