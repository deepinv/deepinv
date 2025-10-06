r"""
Traversing the Perception-Distortion Trade-off
==============================================

This example shows you how to train generative adversarial networks (GANs)
to traverse the Perception-Distortion Trade-off :footcite:p:`blau2018perception`.


"""

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import deepinv as dinv

ORIGINAL_DATA_DIR = dinv.utils.demo.get_data_home()
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Dataset
# ~~~~~~~
# We train the GAN using the MNIST dataset of handwritten digits,
# as also used in :footcite:p:`blau2018perception`.
# For speed we use a subset of the images for training, but you should use the full dataset
# by removing the subset.

class MNISTDataset(datasets.MNIST):
    def __getitem__(self, index):
        return super().__getitem__(index)[0] # return just x
    
train_dataloader = DataLoader(
    Subset(
        MNISTDataset(
            ORIGINAL_DATA_DIR, train=True, transform=transforms.ToTensor(), download=True
        ), indices=range(60000),
    ), batch_size=64, shuffle=True, generator=torch.Generator(device="cpu").manual_seed(0)
)

test_dataloader = DataLoader(
    Subset(
        MNISTDataset(
            ORIGINAL_DATA_DIR, train=False, transform=transforms.ToTensor(), download=True
        ), indices=range(10000),
    ), batch_size=64, shuffle=False
)

def display(x, h=3, w=3):
    if len(x) > h*w:
        x = x[:h*w]
    return torch.cat(torch.chunk(
        torch.cat(torch.chunk(x, h, dim=0), dim=-2),
        w, dim=0
    ), dim=-1)

#dinv.utils.plot(display(next(iter(test_dataloader))), ["Test ground truth images"])

# %%
# Models
# ~~~~~~
# For our reconstructor :math:`\inverse{\cdot}` and the discriminator model :math:`D(\cdot)`
# we use small convolutional architectures used in the WGAN-GP paper :footcite:p:`gulrajani2017improved`.

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=0),  # -> (32, 12, 12)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0), # -> (64, 4, 4)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),                                          # -> 64*4*4 = 1024
            nn.Linear(1024, 1)
        )

    def forward(self, x): # Input: (B, 1, 28, 28)
        return self.net(x) # Output: (B, 1)

class MNISTGenerator(dinv.models.Denoiser):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 4 * 4 * 128),
            nn.BatchNorm1d(4 * 4 * 128),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, *args, **kwargs): # Input: (B, 1, 28, 28)
        return self.decoder(self.encoder(x)) # Output: (B, 1, 28, 28)

# %%
# Losses
# ~~~~~~
# To train the GANs, we must specify the metric in :class:`deepinv.loss.adversarial.DiscriminatorMetric`
# which specifies the GAN "flavour". By default this is uses the mean squared error (i.e. LSGAN :footcite:t:`mao2017least`)
# but here we pass in instead the signed identity which recovers the Wasserstein GAN :footcite:p:`arjovsky2017wasserstein,gulrajani2017improved`.
# 

class WGANMetric(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if (target == 1.).all():
            return -pred.mean()
        elif (target == 0.).all():
            return pred.mean()
        else:
            raise ValueError("Target should be all 0 or all 1.")

metric_gan = dinv.loss.adversarial.DiscriminatorMetric(metric=WGANMetric(), device=device)

# %%
# Training
# ~~~~~~~~

results = {}
for lmbd in tqdm([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]):
    model = dinv.models.ArtifactRemoval(MNISTGenerator(), device=device)
    D = MNISTDiscriminator().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)

    loss = [
        dinv.loss.SupLoss(metric=torch.nn.MSELoss()),
        dinv.loss.adversarial.SupAdversarialLoss(
            D=D, weight_adv=lmbd, device=device, metric_gan=metric_gan,
            optimizer_D=optimizer_D, scheduler_D=scheduler_D, num_D_steps=4,
        ),
    ]

    rng = torch.Generator(device=device).manual_seed(0)
    physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=1., rng=rng))

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        epochs=35,
        losses=loss,
        online_measurements=True,
        save_path=None,
        device=device,
        show_progress_bar=False,
        verbose=False,
    )
    trainer.train()
    trainer.metrics = [ # TODO directly in trainer.test after #777 merged
        loss[0],
        dinv.metric.Metric(lambda x_net, x, *args, **kwargs: loss[1].adversarial_discrim(x, x_net) / lmbd),
    ]

    # Get qualitative results
    results[lmbd] = trainer.test(test_dataloader) #metrics=metrics

    # Get quantitative results
    with torch.no_grad():
        results[lmbd] |= {
            "x": display(x := next(iter(test_dataloader)).to(device)),
            "y": display(y := physics(x)),
            "x_hat": display(model(y, physics))
        }

# %%
# Results
# ~~~~~~~
# We now plot perception vs distortion for the varying :math:`\lambda`,
# where the distortion metric is given by the supervised MSE loss and the
# perceptual metric is given by the supervised adversarial loss. We 
# observe that there is a visible monotonic trade-off.

fig = plt.figure(figsize=(10, 4))

x, y = zip(*[(r["SupLoss"], r["SupAdversarialLoss"]) for r in results.values()])
plt.scatter(x, y)
[plt.annotate(lmbd, (x[i], y[i])) for (i, lmbd) in enumerate(results.keys())]
plt.xlabel("MSE")
plt.ylabel("Wasserstein metric")
fig.savefig("/home/s2558406/models/deepinv/perceptual/results.png")

# %%
# We also plot the reconstructions to visualise the trade-off. Observe that
# increasing lambda gains "sharpness" of the images, but decreases accuracy ("hallucinates").

dinv.utils.plot({
    "x": results[0.01]["x"],
    "y": results[0.01]["y"],
} | {
    f"lmbd={lmbd}": results[lmbd]["x_hat"] for lmbd in results.keys()
}, suptitle="Recon results for varying lambda", save_fn="/home/s2558406/models/deepinv/perceptual/recons.png")