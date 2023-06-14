import torch
from torch.utils.data.dataset import Dataset
import scipy.io as scio


class CTData(Dataset):
    """CT dataset."""

    def __init__(
        self,
        mode="train",
        root_dir="../datasets/CT100_256x256.mat",
        download=True,
        sample_index=None,
    ):
        # the original CT100 dataset can be downloaded from
        # https://www.kaggle.com/kmader/siim-medical-images
        # the images are resized and saved in Matlab.

        if download:
            download_dataset(dataset_name="CT100")

        mat_data = scio.loadmat(root_dir)
        x = torch.from_numpy(mat_data["DATA"])

        if mode == "train":
            self.x = x[0:90]
        if mode == "test":
            self.x = x[90:100, ...]

        self.x = self.x.type(torch.FloatTensor)

        if sample_index is not None:
            self.x = self.x[sample_index].unsqueeze(0)

    def __getitem__(self, index):
        x = self.x[index]
        return x

    def __len__(self):
        return len(self.x)


def ct100_dataloader(train=True, batch_size=1, shuffle=True, num_workers=1):
    return torch.utils.data.DataLoader(
        dataset=CTData(mode="train" if train else "test"),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
