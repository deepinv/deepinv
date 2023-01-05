from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import os
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils import data


class OnlineDataset(Dataset):
    """Generate inverse problems dataset from base signal dataset."""
    def __init__(self, dataset, physics, supervised=True):
        """
        Args:
            dataset (torch.Dataset): base dataset with signals (e.g. mnist dataset).
            physics (dinv.physics or callable): forward operator.
            supervised (boolean): generate supervised pairs (x,y) or unsupervised measurements (y)
        """
        self.dataset = dataset
        self.physics = physics
        self.supervised = supervised

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset.data[idx]
        x = x[0] if isinstance(x, list) else x # get rid of labels
        return x

    def collate(self, batch: list[np.ndarray, int], device: torch.device) -> tuple[torch.Tensor]:
        x, labels = torch.utils.data.default_collate(batch)
        x = x.to('cuda:0')
        if self.physics is not None:
            y = self.physics(x)

        if self.supervised:
            return x, y
        else:
            return y


class HDF5Dataset(data.Dataset):
    """
    Represents a DeepInverse HDF5 dataset which stores measurements and (optionally) associated signals.
    ----------
    file_path
        Path to the folder containing the dataset (one or multiple HDF5 files).
    transform
        PyTorch transform to apply to every data instance (default=None). Only use if the forward operator
        is equivariant to the transform.
    """

    def __init__(self, path, train=True, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.transform = transform

        hd5 = h5py.File(path, 'r')
        if train:
            self.x = hd5['x_train']
            self.y = hd5['y_train']
        else:
            self.x = hd5['x_test']
            self.y = hd5['y_test']

    def __getitem__(self, index):
        # get signal
        x = torch.from_numpy(self.x[index]).type(torch.float)
        y = torch.from_numpy(self.y[index]).type(torch.float)

        return x, y

    def __len__(self):
        return len(self.y)


def generate_dataset(train_dataset, physics, save_dir, test_dataset=None, device='cuda:0', max_datapoints=1e6,
                     dataset_filename='dinv_dataset', batch_size = 32, num_workers=4, supervised=True):
    """
    Args:
        max_datapoints (int): Maximum desired number of datapoints in the dataset. If larger than len(base_dataset),
        the function will use the whole base dataset.
    """
    if os.path.exists(save_dir + dataset_filename):
        print("WARNING: Dataset already exists, this will overwrite the previous dataset.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not (type(physics) in [list, tuple]):
        physics = [physics]
        G = 1
    else:
        G = len(physics)

    n_train = min(len(train_dataset), max_datapoints)
    n_train_g = int(n_train/G)

    n_test = min(len(test_dataset), max_datapoints)
    n_test_g = int(n_test/G)

    for g in range(G):
        hf = h5py.File(f"{save_dir}/{dataset_filename}{g}.h5", 'w')

        hf.attrs['operator'] = physics[g].name

        torch.save(physics[g].state_dict(), f"{save_dir}/physics{g}.pt")

        train_dataloader = DataLoader(Subset(train_dataset, indices=list(range(g*n_train_g, (g+1)*n_train_g))),
                                      batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        if G > 1:
            print(f'Computing train measurement vectors from base dataset of operator {g} out of {G}...')
        else:
            print('Computing train measurement vectors from base dataset...')

        index = 0
        for i, x in enumerate(tqdm(train_dataloader)):
            x = x[0] if isinstance(x, list) else x
            x = x.to(device)

            # choose operator
            y = physics[g](x)

            if i == 0:
                hf.create_dataset('y_train', (n_train_g,) + y.shape[1:], dtype='float')
                if supervised:
                    hf.create_dataset('x_train', (n_train_g,) + x.shape[1:], dtype='float')

            # Add new data to it
            bsize = x.size()[0]
            hf['y_train'][index:index+bsize] = y.to('cpu').numpy()
            if supervised:
                hf['x_train'][index:index+bsize] = x.to('cpu').numpy()
            index = index + bsize

        if test_dataset is not None:
            index = 0
            test_dataloader = DataLoader(
                Subset(train_dataset, indices=list(range(g * n_test_g, (g + 1) * n_test_g))),
                batch_size=batch_size, num_workers=num_workers, pin_memory=True)

            if G > 1:
                print(f'Computing test measurement vectors from base dataset of operator {g} out of {G}...')
            else:
                print('Computing test measurement vectors from base dataset...')

            for i, x in enumerate(tqdm(test_dataloader)):
                x = x[0] if isinstance(x, list) else x
                x = x.to(device)

                # choose operator
                y = physics[g](x)

                if i == 0: # create dict
                    hf.create_dataset('x_test', (n_test_g,) + x.shape[1:], dtype='float')
                    hf.create_dataset('y_test', (n_test_g,) + y.shape[1:], dtype='float')

                # Add new data to it
                bsize = x.size()[0]
                hf['x_test'][index:index+bsize] = x.to('cpu').numpy()
                hf['y_test'][index:index+bsize] = y.to('cpu').numpy()
                index = index + bsize
        hf.close()

    print('Dataset has been saved in ' + save_dir)

    return

