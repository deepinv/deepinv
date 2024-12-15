r"""
Pan-sharpening of satellite images
==================================


"""

import deepinv as dinv

# %%
# Simulate pansharpening measurements (4-channel)
physics = dinv.physics.Pansharpen2((4, 256, 256), factor=4, srf="flat")
dataset = dinv.datasets.NBUDataset("nbu", download=False, return_pan=False)
x = dataset[0].unsqueeze(0)  # just MS of shape 1,4,256,256
y = physics(x)

# Pansharpen with classical
x_hat = physics.A_classical(y)

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

loss = dinv.physics.TensorListModule(
    dinv.loss.MCLoss(),
    dinv.loss.SupLoss()
)
metric = dinv.physics.TensorListModule(
    dinv.metric.PSNR(),
    dinv.metric.PSNR()
)
l = loss(y=y, x=x, x_net=x_net, physics=physics, model=model)
m = metric(x=x, x_net=x_net)

# %%
# Perform pansharpening using raw measurements y ("unsupervised")
physics = dinv.physics.Pansharpen((4, 1024, 1024), factor=4)
dataset = dinv.datasets.NBUDataset("nbu", return_pan=True)
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
