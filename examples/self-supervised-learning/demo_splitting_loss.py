r"""
Self-supervised learning with measurement splitting
===================================================

"""

import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from torchvision import transforms, datasets
from deepinv.models.utils import get_weights_url

torch.manual_seed(0)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# %%
# Define loss
# ~~~~~~~~~~~
# 

loss = dinv.loss.SplittingLoss(split_ratio=0.6, eval_split_output=True)


# %%
# Prepare data
# ~~~~~~~~~~~~
# 

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=".", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root=".", train=False, transform=transform, download=True
)

physics = dinv.physics.Denoising(dinv.physics.PoissonNoise(0.1))

deepinv_datasets_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    device=device,
    save_dir="MNIST",
    train_datapoints=100,
    test_datapoints=10,
)

train_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(path=deepinv_datasets_path, train=False)

train_dataloader = DataLoader(
    train_dataset, shuffle=True
)
test_dataloader = DataLoader(
    test_dataset,  shuffle=False
)



# %%
# Define model
# ~~~~~~~~~~~~
# 
# Load pretrained model trained with 1000 images for 100 epochs. We
# demonstrating training with 100 images for 1 epoch.
# 

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(in_channels=1, out_channels=1, scales=2).to(device)
)
model = loss.adapt_model(model, MC_samples=5)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

file_name = "demo_measplit_mnist_denoising.pth"
url = get_weights_url(model_name="measplit", file_name=file_name)
ckpt = torch.hub.load_state_dict_from_url(
    url, map_location=lambda storage, loc: storage, file_name=file_name
)

model.load_state_dict(ckpt["state_dict"])
optimizer.load_state_dict(ckpt["optimizer"])

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    epochs=1,
    losses=loss,
    optimizer=optimizer,
    device=device,
    train_dataloader=train_dataloader,
    plot_images=False,
    save_path=None,
    verbose=True,
    show_progress_bar=False,
    wandb_vis=False,
)

model = trainer.train()

trainer.plot_images = True
model.MC_samples = 50
trainer.test(test_dataloader)

model.eval_split_output = False
trainer.test(test_dataloader)

