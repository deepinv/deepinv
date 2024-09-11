import sys

sys.path.append("/home/zhhu/workspaces/deepinv/")

import deepinv as dinv
from time import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from deepinv.utils.demo import load_url_image, get_image_url

# img_sizes: list[int] = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
img_sizes: list[int] = [8, 16, 32, 64, 96, 128, 160, 192, 224, 240]
print(img_sizes)
n_repeats = 100
oversampling = 1

url = get_image_url("SheppLogan.png")

# random model
df_random_cpu = pd.DataFrame(
    {
        **{f"img_size_{img_size}": None for img_size in np.array(img_sizes)},
    },
    index=[0],
)

for img_size in tqdm(img_sizes):
    x = load_url_image(
        url=url, img_size=img_size, grayscale=True, resize_mode="resize", device="cpu"
    )
    x_phase = torch.exp(1j * x * torch.pi - 0.5j * torch.pi).to("cpu")

    for i in range(n_repeats):
        physics = dinv.physics.RandomPhaseRetrieval(
            m=int(oversampling * torch.prod(torch.tensor(x.shape))),
            img_shape=(1, img_size, img_size),
            dtype=torch.cfloat,
            device="cpu",
        )
        init_time = time()
        y = physics.forward(x_phase)
        df_random_cpu.loc[i, f"img_size_{img_size}"] = time() - init_time

df_random_cpu.to_csv("df_random_cpu.csv", index=False)

