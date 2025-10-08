r"""
Learned Regularization Functionals
====================================================================================================

In this example, we show how to solve inverse problems using a learned regularizer and the nonmonotonic accelerated proximal gradient algorithm.
We consider denoising, computed tomography and inpainting with the convex ridge regularizer (CRR) (:footcite:t:`goujon2023neural`),
weakly convex ridge regularizer (WCRR) (:footcite:t:`goujon2024learning`) and least squares regularizer (LSR) (see, e.g., :footcite:t:`hurault2021gradient` or :footcite:t:`zou2023deep`).
This example only covers the reconstruction with these regularizers. For the training them in a bilevel regime we refer to :footcite:t:`hertrich2025learning`.

For the reconstructions we sovle the variational problem

    .. math::
        \begin{equation}
        \label{eq:min_prob}
        \tag{1}
        \underset{x}{\arg\min} \quad  \datafid{Ax}{y} + \lambda \reg{x},
        \end{equation}

for a data-fidelity term :math:`\datafid`, forward operator :math:`A` and a learned regularizer :math:`reg` using the
:class:`nonmonotonic accelerated proximal gradient descent <deepinv.optim.NMAPG>` algorithm.
"""

import deepinv as dinv
from deepinv.optim import WCRR, LSR, NMAPG
from torchvision import transforms
from deepinv.utils.demo import load_example
from deepinv.optim import L2, IndicatorL2
from deepinv.physics import Denoising, Tomography, Inpainting, GaussianNoise
import torch
from deepinv.loss.metric import PSNR
from deepinv.utils.plotting import plot

torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
test_img = load_example("CBSD_0010.png", grayscale=False).to(device)
test_img = test_img[:, :, 50:-150, 50:-50]  # make image smaller to run faster
test_img_ct = load_example(
    "SheppLogan.png", img_size=64, resize_mode="resize", grayscale=True, device=device
)
psnr = PSNR()  # fixed range PSNR
psnr_dyn = PSNR(min_pixel=None, max_pixel=None)  # dynamic range PSNR

# %%
# Loading physics, data fidelity term and regularizers
# --------------------------------------------------------------------
# In this example we use the CRR, WCRR and LSR. First, we load the models with the pretrained weights (trained on BSDS500) in deepinv.
# Moreover we load the data fidelity terms and the three different physics operators (Denoising, CT and Inpainting).
#

# color versions
crr = WCRR(
    weak_convexity=0.0, device=device
)  # the CRR is the WCRR with weak convexity 0.0
wcrr = WCRR(weak_convexity=1.0, device=device)
lsr = LSR(device=device)

# grayscale versions
crr_gray = WCRR(
    in_channels=1, weak_convexity=0.0, device=device
)  # the CRR is the WCRR with weak convexity 0.0
wcrr_gray = WCRR(in_channels=1, weak_convexity=1.0, device=device)
lsr_gray = LSR(in_channels=1, device=device)

# data fidelity and physics
data_fidelity_l2 = L2()
data_fidelity_ind = IndicatorL2(0)
physics_denoising = Denoising(noise_model=GaussianNoise(sigma=25 / 255))
physics_inpainting = Inpainting(test_img[0].shape, mask=0.3, device=device)
physics_tomography = Tomography(
    angles=60,
    img_width=test_img_ct.shape[-1],
    circle=False,
    device=device,
    noise_model=GaussianNoise(sigma=0.7),
)

# %%
# Denoising
# --------------------------------------------------------------------
# We start with image denoising. The denosing reconstruction coincides with the proximal operator. This is internally solved using the
# nmAPG, but we only have to call the prox function in this case
#

# create observation
y = physics_denoising(test_img)

# reconstruction via the prox function
recon_crr = crr.prox(y, gamma=0.5)
recon_wcrr = wcrr.prox(y, gamma=0.5)
recon_lsr = lsr.prox(y, gamma=0.5)

psnr_noisy = psnr(y, test_img).item()
psnr_crr = psnr(recon_crr, test_img).item()
psnr_wcrr = psnr(recon_wcrr, test_img).item()
psnr_lsr = psnr(recon_lsr, test_img).item()

print(
    f"Resulting PSNRs:\nNoisy image: {psnr_noisy:.2f}, CRR: {psnr_crr:.2f}, WCRR: {psnr_wcrr:.2f}, LSR: {psnr_lsr:.2f}"
)

plot(
    [test_img, y, recon_crr, recon_wcrr, recon_lsr],
    ["ground truth", "noisy", "CRR", "WCRR", "LSR"],
)

# %%
# Computed Tomography
# --------------------------------------------------------------------
# Next, we consider a CT problem. Here, we directly call the nmAPG for minimizing the variational problem.
# In contrast to the previous example, the images are gray-valued. The data fidelity term is L2.
# Since the problem is smooth, we do not use the proximal mapping in the nmAPG but apply the gradient part to
# both terms (data fidelity and prior).
#

# create observation
y = physics_tomography(test_img_ct)

lmbd = 60  # regularization parameter

# create models
model_crr = NMAPG(data_fidelity_l2, crr_gray, lmbd)
model_wcrr = NMAPG(data_fidelity_l2, wcrr_gray, lmbd)
model_lsr = NMAPG(data_fidelity_l2, lsr_gray, lmbd)

fbp = physics_tomography.A_dagger(y)  # filtered backprojection

# reconstruct
recon_crr = model_crr(y, physics_tomography)
recon_wcrr = model_wcrr(y, physics_tomography)
recon_lsr = model_lsr(y, physics_tomography)

# compute PSNR
psnr_fbp = psnr_dyn(fbp, test_img_ct).item()
psnr_crr = psnr_dyn(recon_crr, test_img_ct).item()
psnr_wcrr = psnr_dyn(recon_wcrr, test_img_ct).item()
psnr_lsr = psnr_dyn(recon_lsr, test_img_ct).item()

print(
    f"Resulting PSNRs:\nFBP: {psnr_fbp:.2f}, CRR: {psnr_crr:.2f}, WCRR: {psnr_wcrr:.2f}, LSR: {psnr_lsr:.2f}"
)
plot(
    [test_img_ct, fbp, recon_crr, recon_wcrr, recon_lsr],
    ["ground truth", "FBP", "CRR", "WCRR", "LSR"],
)

# %%
# Inpainting
# --------------------------------------------------------------------
# Finally, we consider noise-free inpainting with a random mask. Here, the data fidelity term is given by the indicator function.
# Since it is non-smooth, we apply the proximal mapping for the data-fidelity term in the nmAPG.
# The example works again with color images
#

# create observation
y = physics_inpainting(test_img)

lmbd = 1  # regularization parameter

# create models
model_crr = NMAPG(data_fidelity_ind, crr, lmbd, gradient_for_both=False)
model_wcrr = NMAPG(data_fidelity_ind, wcrr, lmbd, gradient_for_both=False)
model_lsr = NMAPG(data_fidelity_ind, lsr, lmbd, gradient_for_both=False)

masked = physics_inpainting.A_dagger(y)  # observation

# reconstruct
recon_crr = model_crr(y, physics_inpainting)
recon_wcrr = model_wcrr(y, physics_inpainting)
recon_lsr = model_lsr(y, physics_inpainting)

# compute PSNR
psnr_masked = psnr(masked, test_img).item()
psnr_crr = psnr(recon_crr, test_img).item()
psnr_wcrr = psnr(recon_wcrr, test_img).item()
psnr_lsr = psnr(recon_lsr, test_img).item()

print(
    f"Resulting PSNRs:\nMasked: {psnr_masked:.2f}, CRR: {psnr_crr:.2f}, WCRR: {psnr_wcrr:.2f}, LSR: {psnr_lsr:.2f}"
)

plot(
    [test_img, y, recon_crr, recon_wcrr, recon_lsr],
    ["ground truth", "masked", "CRR", "WCRR", "LSR"],
)

# %%
# :References:
#
# .. footbibliography::
