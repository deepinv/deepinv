import torch
import torch.nn.functional as F


def shift_equivariance_error(model, x, displacement):
    y1 = model(x).roll(displacement, dims=(2, 3))
    y2 = model(x.roll(displacement, dims=(2, 3)))
    return y1 - y2


def translation_equivariance_error(model, x, displacement):
    B, C, H, W = x.shape
    displacement_w, displacement_h = displacement

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    )

    grid_x = grid_x + (displacement_w / W * 2)
    grid_y = grid_y + (displacement_h / H * 2)

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

    kwargs = {"mode": "bilinear", "padding_mode": "border", "align_corners": False}

    y1 = model(F.grid_sample(x, grid, **kwargs))
    y2 = F.grid_sample(model(x), grid, **kwargs)
    return y1 - y2
