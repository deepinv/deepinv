import numpy as np

import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.unfolded import unfolded_builder

from models.drunet_multi import DRUNet
from utils.dataloaders import get_div2k, get_fastMRI


# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)



train_patch_size = 128

target_size = (train_patch_size, train_patch_size)

div2k_dataset, blur_generator, physics = get_div2k(train_patch_size, device=device)

batch_size = 10
div2k_dataloader = DataLoader(div2k_dataset, batch_size=batch_size, shuffle=True)

mask_dict = blur_generator.step(batch_size=batch_size)
print(mask_dict['filter'].shape)

for i, x in enumerate(div2k_dataloader):  # fastmri returns an image
    if i == 0:
        x = x.to(device)

        y = physics(x, filter=mask_dict['filter'])
        backproj = physics.A_adjoint(y)

        with torch.no_grad():
            out = model(y, physics)

        dinv.utils.plot([x, y, out], titles=["Sample", "y", "out"], save_dir='tmp_imgs')
        break



fastmri_dataset, mask_generator, physics = get_fastMRI(train_patch_size, device=device)

batch_size = 10
fastmri_dataloader = DataLoader(fastmri_dataset, batch_size=batch_size, shuffle=True)

mask_dict = mask_generator.step(batch_size=batch_size)
print(mask_dict['mask'].shape)

for i, batch in enumerate(fastmri_dataloader):  # fastmri returns an image
    if i == 0:
        x, kspace = batch
        print(x.shape, kspace.shape)

        x = x.to(device)*1e4

        y = physics(x, mask=mask_dict['mask'])

        backproj = physics.A_adjoint(y)

        with torch.no_grad():
            out = model(y, physics)

        dinv.utils.plot([x, backproj, out], titles=["Sample", "backproj", "out"], save_dir='fastmri')
        break

