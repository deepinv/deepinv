r"""
Scan-specific zero-shot measurement splitting for MRI
=====================================================

We demonstrate scan-specific self-supervised learning, that is, learning to 
reconstruct MRI scans from a single accelerated sample without ground truth.

Here, we demonstrate training with the :class:`weighted SSDU <deepinv.loss.WeightedSplittingLoss>` :footcite:p:`millard2023theoretical,yaman2020self`.
However, note that any of the :ref:`self-supervised losses <self-supervised-losses>` can be used to do this with varying performance,
for example see the :ref:`example using Equivariant Imaging <sphx_glr_auto_examples_self-supervised-learning_demo_ei.py>` :footcite:p:`chen2021equivariant`.

"""
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import deepinv as dinv

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device=device).manual_seed(0)
rng_cpu = torch.Generator(device="cpu").manual_seed(0)

# %%
# Data
# ----
# First, download a single volume from the FastMRI brain dataset, via HuggingFace.
#
# .. important::
#    By using this dataset, you confirm that you have agreed to and signed the `FastMRI data use agreement <https://fastmri.med.nyu.edu/>`_.

DATA_DIR = dinv.utils.get_data_home() / "fastMRI" / "multicoil_train"
SLICE_DIR = DATA_DIR / "slices"
OUT_DIR = DATA_DIR / "out"
DATA_DIR.mkdir(parents=True, exist_ok=True) ; SLICE_DIR.mkdir(exist_ok=True) ; OUT_DIR.mkdir(exist_ok=True)

dinv.utils.download_example("demo_fastmri_brain_multicoil.h5", DATA_DIR)

# %%
# We use the FastMRI slice dataset provided in DeepInverse to load the volume and return all 16 slices.
# The data is returned in the format `x, y, params` where `params` is a dictionary containing
# the acceleration mask (simulated Gaussian mask with acceleration 6) and the
# estimated coil sensitivity map.
#
# .. note::
#     This loading takes a few seconds per slice, as it must estimate the coil sensitivity map
#     on the fly.

dataset = dinv.datasets.FastMRISliceDataset(
    DATA_DIR,
    slice_index="all",
    transform=dinv.datasets.MRISliceTransform(
        mask_generator=dinv.physics.generator.GaussianMaskGenerator(
            img_size=(256, 256),
            acceleration=6,
            center_fraction=0.08,
            device="cpu",
            rng=rng_cpu,
        ),
        estimate_coil_maps=True,
    ),
)

# %%
# When training with data that is slow to be loaded, it is faster to save the pre-loaded slices:

if not any(SLICE_DIR.iterdir()):
    for i, (x, y, params) in tqdm(enumerate(dataset)):
        torch.save([x, y, params], SLICE_DIR / f"{i}.pt")

# %%
# Then the dataset can be loaded very quickly. We also pre-normalize to bring the data into a more friendly range.
# We also load a rough noise level as a param to be passed into the physics.
# The ground truth is loaded for evaluation later.

def loader(f):
    x, y, params = torch.load(f, weights_only=True)
    return x * 1e5, y * 1e5, params | {"sigma": 1e-5 * 1e5}

dataset = dinv.datasets.ImageFolder(SLICE_DIR, x_path="*.pt", loader=loader)

# %%
# Physics
# -------
# The multicoil physics is defined every easily:

physics = dinv.physics.MultiCoilMRI(device=device, noise_model=dinv.physics.GaussianNoise(0.))

# %%
# Model
# -----
# For the model we use a fairly large MoDL with a UNet backbone, with 12 iterations:

denoiser = dinv.models.UNet(2, 2, scales=5, batch_norm=False)
model = dinv.models.MoDL(denoiser=denoiser, num_iter=12).to(device)

# %%
# Loss
# ----
# We define the weighted SSDU loss by first defining the generator that generates the splitting masks.
# These splitting masks are multiplied with the measurements as per the original paper.
# Finally, the weighted SSDU loss requires knowledge of the original physics generator to define the weight.
#
# .. info::
#     Feel free to use any self-supervised loss you like here!

split_generator = dinv.physics.generator.GaussianMaskGenerator((256, 256), acceleration=2, center_fraction=0., device=device)
mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator((256, 256), split_generator, device=device)
physics_generator = dinv.physics.generator.GaussianMaskGenerator((256, 256), acceleration=6, center_fraction=0.04, rng=rng, device=device)
loss = dinv.loss.mri.WeightedSplittingLoss(mask_generator=mask_generator, physics_generator=physics_generator)

# %%
# Training
# --------
# We train the model using the self-supervised loss. We randomly split the dataset into training and validation for 
# early stopping (up to a maximum of 100 epochs).
# Because the FastMRI ground truth are cropped magnitude root-sum-of-squares reconstructions, we define a helper metric for evaluation later.

def crop(x_net, x):
    """Crop to GT shape then take magnitude."""
    return dinv.utils.MRIMixin().rss(
        dinv.utils.MRIMixin().crop(x_net, shape=x.shape), multicoil=False
    )

class CropPSNR(dinv.metric.PSNR):
    def forward(self, x_net=None, x=None, *args, **kwargs):
        return super().forward(crop(x_net, x), x, *args, **kwargs)

metric = CropPSNR(max_pixel=None)

train_dataset, val_dataset = torch.utils.data.random_split(dataset, (0.8, 0.2), generator=rng_cpu)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    losses=loss,
    metrics=metric,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-6),
    train_dataloader=DataLoader(train_dataset, shuffle=True),
    eval_dataloader=DataLoader(val_dataset),
    epochs=100,
    save_path=None,
    show_progress_bar=True,
    early_stop=True,
    device=device,
)

model = trainer.train()
model.eval()

# %%
# Evaluation
# ----------
# Now that the model is trained, we test the model on 3 samples
# by evaluating the model, plotting and saving the reconstructions and evaluation metrics.

for i in [len(dataset) // 2 - 1, len(dataset) // 2, len(dataset) // 2 + 1]:
    # Load slice
    x, y, params = default_collate([dataset[i]])
    x, y, params = (x, y.to(device), {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for (k, v) in params.items()})
    
    physics.update(**params)

    # Compute baseline reconstructions
    x_adj = physics.A_adjoint(y).detach().cpu()
    x_dag = physics.A_dagger(y).detach().cpu()

    # Evaluate model
    with torch.no_grad():
        x_hat = model(y, physics).detach().cpu()

    dinv.utils.plot({
        "GT": x,
        "Adjoint": crop(x_adj, x),
        "SENSE": crop(x_dag, x),
        "Trained": crop(x_hat, x),
    }, subtitles=[
        "",
        f"{metric(x_adj, x).item():.2f} dB",
        f"{metric(x_dag, x).item():.2f} dB",
        f"{metric(x_hat, x).item():.2f} dB",
    ], save_fn=OUT_DIR / f"result_{i}.png", close=True, show=False)


# %%
# :References:
#
# .. footbibliography::
