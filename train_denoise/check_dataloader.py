import torch
from torch.utils.data import DataLoader

import deepinv as dinv

from utils.dataloaders import get_fastMRI, get_div2k

device = 'cuda' if torch.cuda.is_available() else "cpu"


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

        sampled = physics(x, filter=mask_dict['filter'])
        backproj = physics.A_adjoint(sampled)

        dinv.utils.plot([x, backproj], titles=["Sample", "backproj"], save_dir='div2k')
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

        x = x.to(device)

        sampled = physics(x, mask=mask_dict['mask'])
        backproj = physics.A_adjoint(sampled)

        dinv.utils.plot([x, kspace, backproj], titles=["Sample", "kspace", "backproj"], save_dir='fastmri')
        break



