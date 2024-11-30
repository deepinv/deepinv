
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv
from deepinv.models.utils import get_weights_url
from deepinv.utils.demo import load_dataset, demo_mri_model

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
rng = torch.Generator(device=device).manual_seed(0)

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"

loss = dinv.loss.SplittingLoss(
    split_ratio=0.5, eval_split_input=True,
    #mask_generator=dinv.physics.generator.GaussianSplittingMaskGenerator((2,128,128), 0.6, device=device, rng=rng)
) # SSDU

#loss = [dinv.loss.EILoss(transform=dinv.transform.Rotate()), dinv.loss.MCLoss()]

img_size = 128

transform = transforms.Compose([transforms.Resize(img_size)])

train_dataset = load_dataset("fastmri_knee_singlecoil", transform, train=True, data_dir=BASE_DIR)
test_dataset = load_dataset("fastmri_knee_singlecoil", transform, train=False, data_dir=BASE_DIR)

physics = dinv.physics.MRI(img_size=(img_size, img_size), device=device)
physics_generator = dinv.physics.generator.GaussianMaskGenerator(acceleration=2, img_size=(img_size, img_size), device=device, rng=rng, center_fraction=0.2)

deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    physics_generator=physics_generator,
    device=device,
    save_dir=DATA_DIR,
    train_datapoints=300,
    test_datapoints=30,
)

# Simulate and load random masks
train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True, load_physics_generator_params=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False, load_physics_generator_params=True)

train_dataloader = DataLoader(train_dataset, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=False)

model = demo_mri_model(device=device)
#model = dinv.models.ArtifactRemoval(dinv.models.UNet(2, 2, batch_norm=False, scales=2), device=device).to(device)
model = loss.adapt_model(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=100,
    losses=loss,
    optimizer=optimizer,
    device=device,
    train_dataloader=train_dataloader,
    plot_images=False,
    save_path=None,
    verbose=True,
    show_progress_bar=False,
)

model = trainer.train()

torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, "demo_measplit_fastmri.pth")

trainer.plot_images = True
#trainer.test(test_dataloader)


# Noise2Inverse
#model.eval_split_input = True
trainer.test(test_dataloader)

#model.eval_n_samples = 1
#trainer.test(test_dataloader)



"""model = demo_mri_model(device=device)
loss = [dinv.loss.EILoss(transform=dinv.transform.CPABDiffeomorphism(device=device)), dinv.loss.MCLoss()]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=30,
    losses=loss,
    optimizer=optimizer,
    device=device,
    train_dataloader=train_dataloader,
    plot_images=False,
    save_path=None,
    verbose=True,
    show_progress_bar=False,
    no_learning_method="A_dagger",  # use pseudo-inverse as no-learning baseline
)

model = trainer.train()

torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, "demo_ei_fastmri.pth")

trainer.plot_images = True
trainer.test(test_dataloader)"""