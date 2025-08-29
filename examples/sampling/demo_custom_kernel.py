r"""
Building your custom MCMC sampling algorithm.
=============================================

This code shows how to build your custom sampling kernel. Here we build a preconditioned Unadjusted Langevin
Algorithm (PreconULA) that takes advantage of the singular value decomposition of the forward operator
to accelerate the sampling.


"""

# %%
import torch
from typing import Any
import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils.demo import load_example

# %%
# Load image from the internet
# ----------------------------
#
# This example uses an image of Messi.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

x = load_example("messi.jpg", img_size=32).to(device)

# %%
# Define forward operator and noise model
# ---------------------------------------
#
# We use a 5x5 box blur as the forward operator and Gaussian noise as the noise model.

sigma = 0.001  # noise level
physics = dinv.physics.BlurFFT(
    img_size=(3, 32, 32),
    filter=torch.ones((1, 1, 5, 5), device=device) / 25,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
)


# %%
# Generate the measurement
# ------------------------
# Apply the forward model to generate the noisy measurement.

y = physics(x)

# %%
# Define the sampling iteration
# -----------------------------
#
# In order to define a custom sampling kernel (possibly a Markov kernel which depends on the previous sample),
# we only need to define the iterator which takes the current sample and returns the next sample.
#
# Here we define a preconditioned ULA iterator (for a Gaussian likelihood),
# which takes into account the singular value decomposition
# of the forward operator, :math:`A=USV^{\top}`, in order to accelerate the sampling.
#
# We modify the standard ULA iteration (see :class:`deepinv.sampling.ULAIterator`) defined as
#
# .. math::
#
#        x_{k+1} = x_{k} + \frac{\eta}{\sigma^2} A^{\top}(y-Ax_{k}) +
#        \eta \alpha \nabla \log p(x_{k}) + \sqrt{2\eta}z_{k+1}
#
#
# by using a matrix-valued step size :math:`\eta = \eta_0 VRV^{\top}` where
# :math:`R` is a diagonal matrix with entries :math:`R_{i,i} = \frac{1}{S_{i,i}^2 + \epsilon}`.
# The parameter :math:`\epsilon` is used to avoid numerical issues when :math:`S_{i,i}^2` is close to zero.
# After some algebra, we obtain the following iteration
#
# .. math::
#
#        x_{k+1} = x_{k} + \frac{\eta_0}{\sigma^2}  V R S (U^{\top}y - S V^{\top}x_{k})
#        +\eta_0  \alpha V R V^{\top}\nabla \log p(x_{k}) + \sqrt{2\eta_0}V\sqrt{R}z_{k+1}
#
# We exploit the methods of :class:`deepinv.physics.DecomposablePhysics` to compute the matrix-vector products
# with :math:`V` and :math:`V^{\top}` efficiently. Note that computing the matrix-vector product with :math:`R` and
# :math:`S` is trivial since they are diagonal matrices.
# See :class:`deepinv.sampling.BaseSampling` for more details on how to create new iterators.


class PreconULAIterator(dinv.sampling.SamplingIterator):
    def __init__(self, algo_params):
        super().__init__(algo_params)

    def forward(self, X, y, physics, data_fidelity, prior, iteration) -> dict[str, Any]:
        x = X["x"]
        x_bar = physics.V_adjoint(x)
        y_bar = physics.U_adjoint(y)

        step_size = self.algo_params["step_size"] / (
            self.algo_params["epsilon"] + physics.mask.pow(2)
        )

        noise = torch.randn_like(x_bar)
        sigma2_noise = 1 / data_fidelity.norm
        lhood = -(physics.mask.pow(2) * x_bar - physics.mask * y_bar) / sigma2_noise
        lprior = (
            -physics.V_adjoint(prior.grad(x, self.algo_params["sigma"]))
            * self.algo_params["alpha"]
        )

        return {
            "x": x
            + physics.V(step_size * (lhood + lprior) + (2 * step_size).sqrt() * noise)
        }


# %%
# Define the prior
# ----------------
#
# The score of a distribution can be approximated using a plug-and-play denoiser via the
# :class:`deepinv.optim.ScorePrior` class.
#
# .. math::
#
#           \nabla \log p_{\sigma_d}(x) \approx \frac{1}{\sigma_d^2} \left(D(x) - x\right)
#
# This example uses a simple median filter as a plug-and-play denoiser.
# The hyperparameter :math:`\sigma_d` controls the strength of the prior.

prior = dinv.optim.ScorePrior(denoiser=dinv.models.MedianFilter())

# %%
# Build our sampler
# -------------------
#
# Using our custom iterator, we can build a sampler class by calling :func:`deepinv.sampling.sampling_builder`
# This function returns an instance of :class:`deepinv.sampling.BaseSampling` which takes care of the sampling procedure
# (calculating mean and variance, taking into account sample thinning and burnin iterations, etc),
# providing a convenient interface to the user.

# load Gaussian Likelihood
likelihood = dinv.optim.data_fidelity.L2(sigma=sigma)

iterations = int(1e2) if torch.cuda.is_available() else 10

# shared ULA/ PreconULA params
step_size = 0.5 * (sigma**2)
denoiser_sigma = 0.1

# parameters for PreconULA
params_preconula = {
    "step_size": step_size,
    "sigma": denoiser_sigma,
    "alpha": 1.0,
    "epsilon": 0.01,
}

# build our PreconULA sampler
preconula = dinv.sampling.sampling_builder(
    PreconULAIterator(params_preconula),
    likelihood,
    prior,
    max_iter=iterations,
    burnin_ratio=0.1,
    thinning=1,
    verbose=True,
)

# parameters for ULA
params_ula = {
    "step_size": step_size,
    "sigma": denoiser_sigma,
    "alpha": 1.0,
}

# build our ULA sampler
ula = dinv.sampling.sampling_builder(
    "ULA",
    likelihood,
    prior,
    params_algo=params_ula,
    max_iter=iterations,
    burnin_ratio=0.1,
    thinning=1,
    verbose=True,
)

# %%
# Run sampling algorithms and plot results
# ----------------------------------------
# Each sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean of each algorithm with a simple linear reconstruction.
#
# The preconditioned step size of the new sampler provides a significant acceleration to standard ULA,
# which is evident in the PSNR of the posterior mean.
#
# .. note::
#   The preconditioned ULA sampler requires a forward operator with an easy singular value decomposition
#   (e.g. which inherit from :class:`deepinv.physics.DecomposablePhysics`) and the noise to be Gaussian,
#   whereas ULA is more general.

ula_mean, ula_var = ula.sample(y, physics)

preconula_mean, preconula_var = preconula.sample(y, physics)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"ULA posterior mean PSNR: {dinv.metric.PSNR()(x, ula_mean).item():.2f} dB")
print(
    f"PreconULA posterior mean PSNR: {dinv.metric.PSNR()(x, preconula_mean).item():.2f} dB"
)

# plot results
imgs = [x_lin, x, ula_mean, preconula_mean]
plot(imgs, titles=["measurement", "ground truth", "ULA", "PreconULA"])
