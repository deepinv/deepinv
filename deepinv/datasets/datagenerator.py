from tqdm import tqdm
import os
import h5py
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils import data


class HDF5Dataset(data.Dataset):
    r'''
    Represents a DeepInverse HDF5 dataset which stores measurements and (optionally) associated signals.

    :param str path: Path to the folder containing the dataset (one or multiple HDF5 files).
    :param bool train: Set to ``True`` for training and ``False`` for testing.
    :param torchvision.Transform transform: PyTorch transform to apply to every data instance (default=``None``).
    '''

    def __init__(self, path, train=True, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.transform = transform
        self.unsupervised = False

        hd5 = h5py.File(path, 'r')
        if train:
            if 'x_train' in hd5:
                self.x = hd5['x_train']
            else:
                self.unsupervised = True
            self.y = hd5['y_train']
        else:
            self.x = hd5['x_test']
            self.y = hd5['y_test']

    def __getitem__(self, index):
        y = torch.from_numpy(self.y[index]).type(torch.float)

        x = y
        if not self.unsupervised:
            x = torch.from_numpy(self.x[index]).type(torch.float)

        return x, y

    def __len__(self):
        return len(self.y)


def generate_dataset(train_dataset, physics, save_dir, test_dataset=None, device='cpu', max_datapoints=1e6,
                     dataset_filename='dinv_dataset', batch_size=4, num_workers=0, supervised=True):
    r'''
    This function generates a dataset of measurement pairs (or measurement only if supervised=False) from a baseline dataset
    (e.g. MNIST, ImageNet) using the forward operator provided by the user.

    :param torch.data.Dataset train_dataset: base dataset with images used for generating associated measurements
        via the chosen forward operator. The generated dataset is saved in HD5 format and can be easily loaded using the HD5Dataset class.
    :param deepinv.physics.Physics physics: Forward operator used to generate the measurement data.
        It can be either a single operator or a list of forward operators. In the latter case, the dataset will be assigned evenly across operators.
    :param str save_dir: folder where the dataset and forward operator will be saved.
    :param torch.data.Dataset test_dataset: if included, the function will also generate measurements associated to the test dataset.
    :param torch.device device: which indicates cpu or gpu.
    :param int max_datapoints: Maximum desired number of datapoints in the dataset. If larger than len(base_dataset),
        the function will use the whole base dataset.
    :param str dataset_filename: desired filename of the dataset.
    :param int batch_size: batch size for generating the measurement data (it only affects the speed of the generating process)
    :param int num_workers: number of workers for generating the measurement data (it only affects the speed of the generating process)
    :param bool supervised: Generates supervised pairs (x,y) of measurements and signals.
        If set to ``False``, it will generate a training dataset with measurements only (y) and a test dataset with pairs (x,y)

    '''
    if os.path.exists(os.path.join(save_dir, dataset_filename)):
        print("WARNING: Dataset already exists, this will overwrite the previous dataset.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not (type(physics) in [list, tuple]):
        physics = [physics]
        G = 1
    else:
        G = len(physics)

    max_datapoints = int(max_datapoints)

    n_train = min(len(train_dataset), max_datapoints)
    n_train_g = int(n_train / G)

    if test_dataset is not None:
        n_test = min(len(test_dataset), max_datapoints)
        n_test_g = int(n_test / G)

    hf_paths = []

    for g in range(G):
        hf_path = f"{save_dir}/{dataset_filename}{g}.h5"
        hf_paths.append(hf_path)
        hf = h5py.File(hf_path, 'w')

        hf.attrs['operator'] = physics[g].__class__.__name__

        torch.save(physics[g].state_dict(), f"{save_dir}/physics{g}.pt")

        train_dataloader = DataLoader(Subset(train_dataset, indices=list(range(g * n_train_g, (g + 1) * n_train_g))),
                                      batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=False if device == 'cpu' else True)

        if G > 1:
            print(f'Computing train measurement vectors from base dataset of operator {g + 1} out of {G}...')
        else:
            print('Computing train measurement vectors from base dataset...')

        index = 0
        for i, x in enumerate(tqdm(train_dataloader)):
            x = x[0] if isinstance(x, list) else x
            x = x.to(device)

            # choose operator and generate measurement
            y = physics[g](x)

            if i == 0:
                hf.create_dataset('y_train', (n_train_g,) + y.shape[1:], dtype='float')
                if supervised:
                    hf.create_dataset('x_train', (n_train_g,) + x.shape[1:], dtype='float')

            # Add new data to it
            bsize = x.size()[0]
            hf['y_train'][index:index + bsize] = y.to('cpu').numpy()
            if supervised:
                hf['x_train'][index:index + bsize] = x.to('cpu').numpy()
            index = index + bsize

        if test_dataset is not None:
            index = 0
            test_dataloader = DataLoader(
                Subset(test_dataset, indices=list(range(g * n_test_g, (g + 1) * n_test_g))),
                batch_size=batch_size, num_workers=num_workers, pin_memory=True)

            if G > 1:
                print(f'Computing test measurement vectors from base dataset of operator {g + 1} out of {G}...')
            else:
                print('Computing test measurement vectors from base dataset...')

            for i, x in enumerate(tqdm(test_dataloader)):
                x = x[0] if isinstance(x, list) else x
                x = x.to(device)

                # choose operator
                y = physics[g](x)

                if i == 0:  # create dict
                    hf.create_dataset('x_test', (n_test_g,) + x.shape[1:], dtype='float')
                    hf.create_dataset('y_test', (n_test_g,) + y.shape[1:], dtype='float')

                # Add new data to it
                bsize = x.size()[0]
                hf['x_test'][index:index + bsize] = x.to('cpu').numpy()
                hf['y_test'][index:index + bsize] = y.to('cpu').numpy()
                index = index + bsize
        hf.close()

    print('Dataset has been saved in ' + str(save_dir))

    return hf_paths
