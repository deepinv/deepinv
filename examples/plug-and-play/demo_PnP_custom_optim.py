"""
PnP with custom optimization algorithm (Condat-Vu Primal-Dual)
====================================================================================================

This example shows how to set its own custom optimization algorithm. 
For example, here, we implement the Condat-Vu Primal-Dual algorithm for Single Pixel Camera (SPC) reconstruction.
"""
import deepinv as dinv
from pathlib import Path
import torch
from deepinv.models import DnCNN
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_image
from deepinv.utils.plotting import plot, plot_curves
from deepinv.optim.optim_iterators import OptimIterator, fStep, gStep

# %%
# Define the custom optimization algorithm as a subclass of OptimIterator,
# along with the corresponding custom fStepCV (subclass of fStep) and gStepCV (subclass of gStep) modules.
# ----------------------------------------------------------------------------------------
#


class CVIteration(OptimIterator):
    r"""
    Single iteration of Condat-Vu Primal-Dual.

    .. math::
        \begin{equation*}
        \begin{aligned}
        x_{k+1} &= \operatorname{prox}_{\tau g}(x_k-\tau A^\top u_k) \\
        z_k &= 2Ax_{k+1}-x_k\\
        u_{k+1} &= \operatorname{prox}_{\sigma f^*}(z_k) \\
        \end{aligned}
        \end{equation*}

    where :math:`f^*` is the Fenchel-Legendre conjugate of :math:`f`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.g_step = gStepCV(**kwargs)
        self.f_step = fStepCV(**kwargs)

    def forward(self, X, cur_prior, cur_params, y, physics):
        r"""
        Single iteration of the PD algorithm.

        :param dict X: Dictionary containing the current iterate and the estimated cost.
        :param dict cur_prior: dictionary containing the prior-related term of interest, e.g. its proximal operator or gradient.
        :param dict cur_params: dictionary containing the current parameters of the model.
        :param torch.Tensor y: Input data.
        :param deepinv.physics physics: Instance of the physics modeling the data-fidelity term.
        :return: Dictionary `{"est": (x, ), "cost": F}` containing the updated current iterate and the estimated current cost.
        """
        x_prev, u_prev = X["est"]

        x = self.g_step(x_prev, physics.A_adjoint(u_prev), cur_prior, cur_params)
        u = self.f_step(physics.A(2 * x - x_prev), u_prev, y, cur_params)

        F = self.F_fn(x, cur_params, y, physics) if self.has_cost else None

        return {"est": (x, u), "cost": F}


class fStepCV(fStep):
    r"""
    PD fStep module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, Ax_cur, u, y, cur_params):
        r"""
        Single PD iteration step on the data-fidelity term :math:`f`.

        :param torch.Tensor Ax_cur: Current iterate :math:`2Ax_{k+1}-x_k`
        :param torch.Tensor u: Current iterate :math:`u_k`.
        :param torch.Tensor y: Input data.
        :param dict cur_params: Dictionary containing the current fStep parameters (keys `"stepsize"` and `"lambda"`).
        """
        v = u + cur_params["stepsize"] * Ax_cur
        return v - cur_params["stepsize"] * self.data_fidelity.prox_d(
            v, y, 1 / (cur_params["stepsize"] * cur_params["lambda"])
        )


class gStepCV(gStep):
    r"""
    PD gStep module.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, Atu, cur_prior, cur_params):
        r"""
        Single iteration step on the prior term :math:`g`.

        :param torch.Tensor x: Current iterate :math:`x_k`.
        :param torch.Tensor Atu: Current iterate :math:`A^\top u_k`.
        :param dict cur_prior: Dictionary containing the current prior.
        :param dict cur_params: Dictionary containing the current gStep parameters (keys `"prox_g"`, `"stepsize"` and `"g_param"`).
        """
        return cur_prior.prox(
            x - cur_params["stepsize"] * Atu,
            cur_params["stepsize"],
            cur_params["g_param"],
        )


# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"


# %%
# Load base image datasets and degradation operators.
# ----------------------------------------------------------------------------------------


# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# Set up the variable to fetch dataset and operators.
method = "PnP"
dataset_name = "set3c"
img_size = 256 if torch.cuda.is_available() else 64
url = "https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fdatasets&files=barbara.jpeg"
x = load_image(
    url=url, img_size=img_size, grayscale=True, resize_mode="resize", device=device
)
operation = "single_pixel"


# %%
# Generate a dataset of blurred images and load it.
# --------------------------------------------------------------------------------
# We use the BlurFFT class from the physics module to generate a dataset of blurred images.


noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
n_channels = 1  # 3 for color images, 1 for gray-scale images
physics = dinv.physics.SinglePixelCamera(
    m=100,
    img_shape=(1, 64, 64),
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0


# %%
# Set up the PnP algorithm to solve the inverse problem.
# --------------------------------------------------------------------------------
# We use the Proximal Gradient Descent optimization algoritm.
# The algorithm alternates between a denoising step and a gradient descent step.
# The denoising step is performed by a DNCNN pretrained denoiser :class:`deepinv.models.dncnn`.

# Logging parameters
verbose = True
plot_metrics = True  # compute performance and convergence metrics along the algorithm, curved saved in RESULTS_DIR

# Set up the PnP algorithm parameters : the `stepsize`, `g_param` the noise level of the denoiser and `lambda` the regularization parameter. The following parameters are chosen arbitrarily.
params_algo = {"stepsize": 1.0, "g_param": noise_level_img, "lambda": 0.1}
max_iter = 200
early_stop = True  # stop the algorithm when convergence is reached

# Select the data fidelity term
data_fidelity = L2()

# Specify the denoising prior
denoiser = DnCNN(
    in_channels=n_channels,
    out_channels=n_channels,
    pretrained="download",
    train=False,
    device=device,
)
prior = PnP(denoiser=denoiser)

# instantiate the algorithm class to solve the IP problem.
algo = CVIteration(data_fidelity=data_fidelity, F_fn=None, has_cost=False)
model = optim_builder(
    iteration=algo,
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    verbose=verbose,
    params_algo=params_algo,
    return_metrics=plot_metrics,
)

# %%
# Evaluate the model on the problem and plot the results.
# --------------------------------------------------------------------

y = physics(x)
x_lin = physics.A_adjoint(y)

# run the model on the problem. When `return_metrics` is set to True, the model requires the ground-truth clean image ``x_gt`` and returns the output and the metrics computed along the iterations.
x_model, metrics = model(y, physics, x_gt=x)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_lin):.2f} dB")
print(f"Model reconstruction PSNR: {dinv.utils.metric.cal_psnr(x, x_model):.2f} dB")

# plot results
imgs = [x, x_lin, x_model]
plot(imgs, titles=["GT", "Linear", "Recons."], show=True)

# plot convergence curves
if plot_metrics:
    plot_curves(metrics, save_dir=RESULTS_DIR / "curves", show=True)
