import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from deepinv.models.denoiser import Denoiser
from deepinv.training_utils import test

# num_workers = 4  # set to 0 if using small cpu
num_workers = 0  # set to 0 if using small cpu
problem = "denoising"
G = 1
# PREVIOUS
# denoiser_name = 'GSDRUNet'
# ckpt_path = '../checkpoints/GSDRUNet.ckpt'
# NEW: update the following dictionnary with parameters for GSDRUNet appropriately
n_channels = 3
pretrain = True
train = False
model_spec = {
    "name": "gsdrunet",
    "args": {
        "in_channels": n_channels + 1,
        "out_channels": n_channels,
        "ckpt_path": "../checkpoints/GSDRUNet.ckpt",
        "pretrain": pretrain,
        "train": train,
        "device": dinv.device,
    },
}
# pnp_algo = 'HQS'
batch_size = 1
dataset = "set3c"
dataset_path = "../../datasets/set3c"
dir = f"../datasets/{dataset}/{problem}/G{G}/"
noise_level_img = 20 / 255
lamb = 0.1
stepsize = 1 / lamb
sigma_k = 2.0
sigma = sigma_k * noise_level_img

physics = []
dataloader = []

for g in range(G):
    if problem == "CS":
        p = dinv.physics.CompressedSensing(
            m=300, img_shape=(1, 28, 28), device=dinv.device
        )
    elif problem == "onebitCS":
        p = dinv.physics.CompressedSensing(
            m=300, img_shape=(1, 28, 28), device=dinv.device
        )
        p.sensor_model = lambda x: torch.sign(x)
    elif problem == "inpainting":
        p = dinv.physics.Inpainting(
            tensor_size=(1, 28, 28), mask=0.5, device=dinv.device
        )
    elif problem == "denoising":
        p = dinv.physics.Denoising(sigma=0.2)
    elif problem == "blind_deblur":
        p = dinv.physics.BlindBlur(kernel_size=11)
    elif problem == "deblur":
        p = dinv.physics.Blur(
            dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0),
            device=dinv.device,
        )
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    p.load_state_dict(torch.load(f"{dir}/physics{g}.pt", map_location=dinv.device))
    physics.append(p)

    dataset = dinv.datasets.HDF5Dataset(path=f"{dir}/dinv_dataset{g}.h5", train=False)
    dataloader.append(
        DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
    )

denoiser = Denoiser(model_spec=model_spec)
model = lambda x, physics: denoiser(x, sigma)
plot = True

test(
    model=model,
    test_dataloader=dataloader,
    physics=physics,
    plot=plot,
    device=dinv.device,
    save_img_path="results/example_denoising.png",
    plot_input=True,
)
