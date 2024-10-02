import torch
from models.blocks import ConvNextBlock

# from models.unext_v2 import UNeXt
from models.unext_wip import UNeXt

device = "cuda" if torch.cuda.is_available() else "cpu"

# set seed
torch.manual_seed(0)

x = torch.rand(1, 3, 128, 128).to(device)
model_test = "unext"

if model_test == "unext":
    # net = UNeXt(in_channels=3, out_channels=3, scales=4, device=device, residual=True).to(device)
    # net = UNeXt(in_channels=3, out_channels=3, device=device, conv_type='next', pool_type='next', replk=1).to(device)
    net = UNeXt(
        in_channels=3, out_channels=3, device=device, replk=1, pool_type="next"
    ).to(device)
    sigma = 0.1
elif model_test == "convnext":
    net = ConvNextBlock(in_channels=3, out_channels=3)
    sigma = torch.rand(1, 512)

# print(net)
print(
    "The model has ",
    sum(p.numel() for p in net.parameters() if p.requires_grad),
    "parameters",
)

# sigma = torch.tensor([sigma], device=device)
y = net(x, sigma)
print("shape y", y.shape)
print("shape x", x.shape)

sx = 1
sy = 1

x1 = torch.roll(x, dims=(2, 3), shifts=(sx, sy))
y1 = net(x1, sigma)
y1 = torch.roll(y1, dims=(2, 3), shifts=(-sx, -sy))

error = torch.norm(y - y1)

print("shift-equivariance error:", error.item())

error = torch.norm(net(1.5 * x1, 1.5 * sigma) - (1.5 * net(x1, sigma)))
print("scale equivariance error", error.item())


error = torch.norm(net(1.5 * x1 + 1, 1.5 * sigma) - (1.5 * net(x1, sigma) + 1))
print("normalization equivariance error", error.item())


# print('parameters', sum(p.numel() for p in net.parameters() if p.requires_grad))
