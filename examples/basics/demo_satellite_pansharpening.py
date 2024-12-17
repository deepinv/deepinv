r"""
Pan-sharpening of satellite images
==================================


"""

import deepinv as dinv

# %%
# Simulate pansharpening measurements (4-channel)
physics = dinv.physics.Pansharpen((4, 256, 256), factor=4, srf="flat")
dataset = dinv.datasets.NBUDataset(".", download=False, return_pan=False)
x = dataset[0].unsqueeze(0)  # just MS of shape 1,4,256,256
y = physics(x)

# Pansharpen with Brovey's method
x_hat = physics.A_dagger(y)

# Pansharpen with model
model = dinv.models.PanNet(hrms_shape=(4, 256, 256))
x_net = model(y, physics)

dinv.utils.plot(
    [
        x[:, :3],
        y[0][:, :3],
        y[1],
        x_hat[:, :3],
        x_net[:, :3],
    ],
    titles=["x MS", "y MS", "y PAN", "x_hat (classical)", "x_net (PanNet)"],
)

# Evaluate performance ("supervised")
qnr = dinv.metric.QNR()
sam = dinv.metric.distortion.SpectralAngleMapper()
ergas = dinv.metric.distortion.ERGAS(factor=4)
print(qnr(x_hat, x=None, y=y, physics=physics), sam(x_hat, x), ergas(x_hat, x))

# Example training
loss = dinv.loss.StackedPhysicsLoss([
    dinv.loss.MCLoss(),
    dinv.loss.SureGaussianLoss(0.05)
])

from torch.optim import Adam
from torch.utils.data import DataLoader
trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=Adam(model.parameters()),
    losses=loss,
    train_dataloader=DataLoader(dataset),
    online_measurements=True,
    
)

# %%
# Perform pansharpening using raw measurements y ("unsupervised")
physics = dinv.physics.Pansharpen((4, 1024, 1024), factor=4)
dataset = dinv.datasets.NBUDataset(".", return_pan=True)
y = dataset[0].unsqueeze(0)  # MS (1,4,256,256), PAN (1,1,1024,1024)

dinv.utils.plot(
    [
        y[0][:, :3],
        y[1],
        physics.A_classical(y)[:, :3],  # shape (1,4,1024,1024)
        physics.downsampling.A_adjoint(y[0])[:, :3],
        physics.A_adjoint(y)[:, :3],
        physics.A_dagger(y)[:, :3],
    ],
    titles=[
        "Input MS",
        "Input PAN",
        "Brovey reconstruction",
        "Downsampling adjoint",
        "pansharpening adjoint",
        "pansharpening dagger",
    ],
    dpi=1200,
)

# Evaluate performance - note we can only use QNR as we have no GT
qnr = dinv.metric.QNR()
print(qnr(x_net=physics.A_classical(y), x=None, y=y, physics=physics))
