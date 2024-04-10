r"""
Random phase retrieval and reconstruction methods.
===================================================

In this example, we will show how to create a random phase retrieval operator and use it to generate measurements. We then show how to use gradient descent and spectral methods to reconstruct the image given theses measurements.

"""

import matplotlib.pyplot as plt
import torch

import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.utils.demo import load_url_image, get_image_url

# %%
# Load image from the internet
# ----------------------------
#
# This example uses an image of the CBSD68 dataset.

# Set global random seed to ensure reproducibility.
torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = get_image_url("CBSD_0010.png")
x = load_url_image(url, grayscale=False).to(device)
x = torch.tensor(x, device=device, dtype=torch.float)
# Downsample the image to 32x32 pixels
x = torch.nn.functional.interpolate(x, size=(32, 32))

img_shape = x.shape[1:]

# %%
# Visualization
# ---------------------------------------
#
# We use the customized plot() function in deepinv to visualize the original image.
plot(x, title="Original image")

# %%
# Signal Construction
# ---------------------------------------
# We use the original image as the phase information for the complex signal. The original value range is [0, 1], so we scale it to cover the full phase range[0, 2*pi].
x_phase = torch.exp(1j * 2 * torch.pi * x)

# For phase retrieval, the signal should be complex-valued.
print(x_phase.dtype)

# Every element of the signal should have unit norm.
assert torch.allclose(x_phase.real**2 + x_phase.imag**2, torch.tensor(1.0))

# %%
# Measurements Generation
# ---------------------------------------
# Create a random phase retrieval operator with an
# oversampling ratio (measurements/signals) of 10
m = 10 * torch.prod(torch.tensor(img_shape))
print(f"Number of measurements: {m}")

physics = dinv.physics.RandomPhaseRetrieval(
    m=m,
    img_shape=img_shape,
)

y = physics(x_phase)

# %%
# Reconstruction with Gradient Descent
# ---------------------------------------
# We use the function :class:`deepinv.optim.AmplitudeLoss` as loss function,
# and the class :class:`deepinv.optim.optim_iterators.GDIteration` as the optimizer.
loss_fn = dinv.optim.AmplitudeLoss()
iterator = dinv.optim.optim_iterators.GDIteration()
# Parameters for the optimizer, including stepsize and regularization coefficient.
optim_params = {"stepsize": 0.1, "lambda": 1.0, "g_param": []}
num_iter = 500

# Initial guess
x_est = torch.rand_like(x_phase)

loss_hist = []

for _ in range(num_iter):
    res = iterator(
        {"est": (x_est,), "cost": 0},
        cur_data_fidelity=loss_fn,
        cur_prior=dinv.optim.Zero(),
        cur_params=optim_params,
        y=y,
        physics=physics,
    )
    x_est = res["est"][0]
    loss_hist.append(loss_fn(x_est, y, physics))

print("loss_init:", loss_hist[0])
print("loss_final:", loss_hist[-1])
# Plot the loss curve
plt.plot(loss_hist)
plt.yscale("log")
plt.title("loss curve without spectral")
plt.show()
# Check the average difference between the estimated and the original signal.
torch.mean(x_est - x_phase)

# %%
# Visualization between Original Image and Reconstruction
# -----------------------------------------------------------
# Reconstruct the image using the estimated phase.
# We can use `torch.angle` to extract the phase information. With the range of the returned value being [-pi, pi], we first normalize it to be [0, 1].
x_recon = torch.angle(x_est) / (2 * torch.pi) + 0.5
# A good reconstruction should be only a global phase shift away from the original signal, i.e., `x_recon - x` is constant. We first make sure all reconstruction values are above the corresponding original values
x_recon = torch.where(x_recon < x, x_recon + 1, x_recon)
# Subtract the global phase shift
x_recon = x_recon - (x_recon.max() - x.max())
# Now the reconstruction should be close to the signal
print(torch.allclose(x_recon, x, rtol=1e-5))
plot([x, x_recon], titles=["Signal", "Reconstruction"])

# %%
# Reconstruction with Gradient Descent and Spectral Methods
# ---------------------------------------------------------------
# Spectral methods :class:`deepinv.optim.optim_iterators.SMIteration` offers a good intialization for the estimated signals on which a normal gradient descent algorithm can be run.
x_est = torch.rand_like(x_phase)
x_est = x_est / torch.norm(x_est.flatten())

# Create the spectral methods optimizer
spectral = dinv.optim.optim_iterators.SMIteration()

diff_hist = []
# :class:`deepinv.optim.optim_iterators.SMIteration` uses power iteration to find the principal eigenvector. We may run multiple iterations to arrive on a better initialization.
for _ in range(50):
    x_next = spectral(x_est, dinv.optim.Zero(), optim_params, y, physics)
    diff_hist.append(torch.norm(x_next - x_est))
    x_est = x_next

x_est = x_est.reshape(x_phase.shape)

# A classical gradient descent algorithm can then be run on the optimized guess.
loss_hist = []
for _ in range(num_iter):
    res = iterator(
        {"est": (x_est,), "cost": 0},
        cur_data_fidelity=loss_fn,
        cur_prior=dinv.optim.Zero(),
        cur_params=optim_params,
        y=y,
        physics=physics,
    )
    x_est = res["est"][0]
    loss_hist.append(loss_fn(x_est, y, physics))

# Plot the loss curve
print("loss_init:", loss_hist[0])
print("loss_final:", loss_hist[-1])
plt.plot(loss_hist)
plt.yscale("log")
plt.title("loss curve with spectral")
plt.show()
# Check the average difference between the estimated and the original signal.
torch.mean(x_est - x_phase)

# %%
# Visualization between Original Image and Reconstruction
# ------------------------------------------------------------
# Reconstruct the image using the estimated phase.
# We can use `torch.angle` to extract the phase information. With the range of the returned value being [-pi, pi], we first normalize it to be [0, 1].
x_recon = torch.angle(x_est) / (2 * torch.pi) + 0.5
# A good reconstruction should be only a global phase shift away from the original signal, i.e., `x_recon - x` is constant. We first make sure all reconstruction values are above the corresponding original values
x_recon = torch.where(x_recon < x, x_recon + 1, x_recon)
# Subtract the global phase shift
x_recon = x_recon - (x_recon.max() - x.max())
# Now the reconstruction should be close to the signal
print(torch.allclose(x_recon, x, rtol=1e-5))
plot([x, x_recon], titles=["Signal", "Reconstruction"])
