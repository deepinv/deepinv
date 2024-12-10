import deepinv as dinv

# %%
# Simulate pansharpening measurements (4-channel)
physics = dinv.physics.Pansharpen((4, 256, 256), factor=4, noise_gray=None, srf="flat")
dataset = dinv.datasets.NBUDataset("nbu", download=True, return_pan=False)
y = dataset[0].unsqueeze(0) # just MS of shape 1,4,256,256

dinv.utils.plot([
    y[:, :3],
    physics(y)[0][:, :3],
    physics(y)[1]
], titles=[
    "x MS",
    "y MS",
    "y PAN"
])

# %%
# Perform pansharpening using raw measurements y
physics = dinv.physics.Pansharpen((4, 1024, 1024), factor=4)
dataset = dinv.datasets.NBUDataset("nbu", return_pan=True)
y = dataset[0].unsqueeze(0) # MS (1,4,256,256), PAN (1,1,1024,1024)

dinv.utils.plot([
    y[0][:, :3],
    y[1],
    physics.A_classical(y)[:, :3], # shape (1,4,1024,1024)
    physics.downsampling.A_adjoint(y[0])[:, :3],
    physics.A_adjoint(y)[:, :3],
    physics.A_dagger(y)[:, :3],
], titles=[
    "Input MS",
    "Input PAN",
    "Brovey reconstruction",
    "Downsampling adjoint",
    "pansharpening adjoint",
    "pansharpening dagger"
], dpi=1200)

# Evaluate performance
qnr = dinv.metric.QNR()

print(qnr(
    x_net=physics.A_classical(y),
    x=None,
    y=y,
    physics=physics
))