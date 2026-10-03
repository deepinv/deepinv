import deepinv as dinv
import kornia
import nevergrad as ng
import glob
import os
import math

class RawDIV2K(dinv.datasets.ImageDataset):
    def __init__(self, root: str, *, upsampling_rate: int):
        super().__init__()

        x_paths = glob.glob(f"{root}/DIV2K_valid_HR/*.png")
        y_paths = glob.glob(f"{root}/DIV2K_valid_LR_bicubic/X{upsampling_rate}/*.png")

        path_pairs = []
        for x_path in x_paths:
            filename = os.path.basename(x_path)
            identifier = filename[:-4]
            matching_path = f"DIV2K/DIV2K_valid_LR_bicubic/X{upsampling_rate}/{identifier}x{upsampling_rate}.png"
            try:
                y_paths.remove(matching_path)
                path_pairs.append((x_path, matching_path))
            except ValueError:
                pass

        if 2 * len(path_pairs) != len(x_paths) + len(path_pairs) + len(y_paths):
            raise RuntimeError(f"Some images could not be paired: {2 * len(path_pairs)} paired images out of {len(x_paths) + len(path_pairs) + len(y_paths)} total images.")

        self._path_pairs = path_pairs

    def __len__(self):
        return len(self._path_pairs)

    def __getitem__(self, idx):
        x_path, y_path = self._path_pairs[idx]
        x = kornia.io.load_image(x_path, desired_type=kornia.io.ImageLoadType.RGB32, device="cpu")
        y = kornia.io.load_image(y_path, desired_type=kornia.io.ImageLoadType.RGB32, device="cpu")
        return x, y


dataset = RawDIV2K("DIV2K", upsampling_rate=2)

def objective_fn(params):
    cls_name = params["cls_name"]
    antialiasing = params["antialiasing"]
    kernel = params["kernel"]
    upsampling_rate = params["upsampling_rate"]
    padding = params["padding"]
    cls = getattr(dinv.physics, cls_name)
    if cls_name == "Downsampling":
        if kernel not in ["valid", "circular", "replicate", "reflect"]:
            return float("inf")
        kwargs = {
            "factor": upsampling_rate,
            "filter": kernel,
            "padding": padding,
        }
    elif cls_name == "DownsamplingMatlab":
        if kernel not in ["cubic", "gaussian"]:
            return float("inf")
        kwargs = {
            "factor": upsampling_rate,
            "kernel": kernel,
            "padding": "reflect",
            "antialiasing": antialiasing
        }
    else:
        raise ValueError(f"Unsupported class name {cls_name}.")

    physics = cls(**kwargs)

    psnrs = []
    for x, y in dataset:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        y_hat = physics(x)

        if y_hat.shape != y.shape:
            return float("inf")
        else:
            psnr = dinv.metric.PSNR(min_pixel=0.0, max_pixel=1.0)(y_hat, y).item()

        psnrs.append(psnr)
        break

    objective = sum(psnrs) / len(psnrs)
    return - objective

param = ng.p.Dict(
    cls_name=ng.p.Choice(["Downsampling", "DownsamplingMatlab"]),
    upsampling_rate=ng.p.Choice([2, 3, 4]),
    kernel=ng.p.Choice(["cubic", "gaussian"]),
    antialiasing=ng.p.Choice([ True, False ]),
    padding=ng.p.Choice(["valid", "circular", "replicate", "reflect"])
)
optimizer = ng.optimizers.TwoPointsDE(parametrization=param, budget=100, num_workers=1)
recommendation = optimizer.minimize(objective_fn)

parameters = recommendation.value
objective = - recommendation.loss
print(f"Best parameters:")
print(parameters)
print(f"Measurement PSNR: {objective:.1f} dB")
