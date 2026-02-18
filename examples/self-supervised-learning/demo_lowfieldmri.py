r"""
Low-field MRI denoising without ground truth
============================================

We demonstrate self-supervised (blind) denoising of a low-field MRI scan without ground truth data.

In low-field MRI, images have high noise due to the fixed permanent magnet, and acquiring clean reference images is often impossible.
One could average the images over multiple repetitions, but if there's any patient motion, the image would be blurry.

Here, we fine-tune the Reconstruct Anything Model (:class:`deepinv.models.RAM`) :footcite:p:`terris2025reconstruct` on a single noisy scan using the self-supervised
:class:`Recorrupted2Recorrupted loss <deepinv.loss.R2RLoss>` :footcite:p:`pang2021recorrupted`.

Play around with different self-supervised denoising losses (see :ref:`self-supervised-losses`)!

"""

import torch
import deepinv as dinv

device = dinv.utils.get_device()

# %%
# Load data
# ---------
# We load a single low-field T1 MRI scan (Oper 0.3T) from the M4Raw dataset :footcite:p:`lyu2023m4raw`, which contains multiple repetitions
# of the same scan with inter-scan motion. We use the first repetition as our noisy measurement,
# and average all three repetitions to create a "reference" (which is still blurry due to motion).
#


def open_m4raw(fname: str) -> torch.Tensor:
    """Load M4Raw slice"""
    x = dinv.io.load_ismrmd(fname, data_slice=8).unsqueeze(
        0
    )  # Load middle slice, shape 12NHW
    x = dinv.utils.MRIMixin().kspace_to_im(x)  # Convert to image space
    x = dinv.utils.MRIMixin().rss(
        x, multicoil=True
    )  # Root-sum-square following original paper, shape 11HW
    x = dinv.utils.normalize_signal(x, mode="min_max")  # Normalise to 0-1
    return x


DATA_DIR = dinv.utils.get_data_home() / "m4raw" / "motion"
DATA_DIR.mkdir(parents=True, exist_ok=True)

dinv.utils.download_example("demo_m4raw_inter-scan_motion_0.h5", DATA_DIR)
dinv.utils.download_example("demo_m4raw_inter-scan_motion_1.h5", DATA_DIR)
dinv.utils.download_example("demo_m4raw_inter-scan_motion_2.h5", DATA_DIR)

y = open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_0.h5")

x = torch.cat(
    [
        open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_0.h5"),
        open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_1.h5"),
        open_m4raw(DATA_DIR / "demo_m4raw_inter-scan_motion_2.h5"),
    ]
).mean(
    dim=0, keepdim=True
)  # Average 3 repetitions

# %%
# Since the data is raw, we estimate the noise level in both the single scan and the averaged scan using patch covariance.
# Note that the averaged scan has lower noise as expected, but observe the motion blurring!

noise_estimator = dinv.models.PatchCovarianceNoiseEstimator()

dinv.utils.plot(
    {"Noisy scan": y, "3x rep, averaged": x},
    subtitles=[
        f"sigma: {noise_estimator(y).item():.4f}",
        f"sigma: {noise_estimator(x).item():.4f}",
    ],
)

# %%
# Physics
# -------
# We define a simple denoising physics with Gaussian noise matching the estimated noise level.

sigma = noise_estimator(y)
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma=sigma.to(device)))

# %%
# Zero-shot reconstruction
# ------------------------
# First, we apply the pre-trained RAM model as a zero-shot denoiser without any training.

model = dinv.models.RAM(device=device)
with torch.no_grad():
    x_net = model(y.to(device), physics)

# %%
# Self-supervised fine-tuning
# ----------------------------
# We create a dataset from the single noisy scan and fine-tune RAM using the R2R loss.
# See the :ref:`dedicated R2R example <sphx_glr_auto_examples_self-supervised-learning_demo_r2r_denoising.py>` for more.
#
# .. note::
#     We train for 50 epochs on GPU. For faster execution on CPU, set epochs to a smaller value.

dataset = dinv.datasets.TensorDataset(y=y, params={"sigma": sigma})

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    train_dataloader=torch.utils.data.DataLoader(dataset, batch_size=1),
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-5),
    epochs=1 if str(device) == "cpu" else 50,
    losses=dinv.loss.R2RLoss(noise_model=None),
    metrics=None,
    device=device,
    save_path=None,
)
model = trainer.train()
model = model.eval()

# %%
# Evaluation
# ----------
# We see that the zero-shot recon is denoised, but details are blurry. The fine-tuned recon has clearer details, which is both less noisy
# than than the input data, but less blurry than the averaged image.

with torch.no_grad():
    x_ft = model(y.to(device), physics)

dinv.utils.plot(
    {
        "Noisy scan": y,
        "3x rep, averaged": x,
        "Zero-shot RAM": x_net,
        "Fine-tuned RAM": x_ft,
    },
)

# %%
# :References:
#
# .. footbibliography::
