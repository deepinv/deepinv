import hashlib
from typing import Any, Callable
import os

from PIL import Image
import torch

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_tarball,
)


class LsdirHR(torch.utils.data.Dataset):
    """Dataset for `LSDIR <https://data.vision.ee.ethz.ch/yawli/>`_.

    A large-scale dataset for image restoration tasks such as image super-resolution (SR),
    image denoising, JPEG deblocking, deblurring, and demosaicking, and real-world SR.


    **Raw data file structure:** ::

        self.root --- 0001000 --- 0000001.png
                   |           |
                   |           -- 0001000.png
                   |  ...
                   |
                   -- 0085000 --- 0084001.png
                   |           |
                   |           -- 0084991.png
                   -- shard-00.tar.gz
                   |  ...
                   -- shard-16.tar.gz
                   |
                   -- val1 --- HR --- val --- 0000001.png
                   |        -- X2          |
                   |        -- X3          -- 0000250.png
                   |        -- X4
                   -- val1.tar.gz

    .. warning::
        The official site hosting the dataset is unavailable : https://data.vision.ee.ethz.ch/yawli/.
        Thus the download argument isn't working for now.

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param str mode: Select a split of the dataset between 'train' or 'val'. Default at 'train'.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional)  A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet ::

            from deepinv.datasets import LsdirHR
            val_dataset = LsdirHR(root="Lsdir", mode="val", download=True)  # download raw data at root and load dataset
            print(val_dataset.verify_split_dataset_integrity())             # check that raw data has been downloaded correctly
            print(len(val_dataset))                                         # check that we have 250 images

    """

    archive_urls = {
        "train": {
            "shard-00.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-00.tar.gz",
            "shard-01.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-01.tar.gz",
            "shard-02.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-02.tar.gz",
            "shard-03.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-03.tar.gz",
            "shard-04.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-04.tar.gz",
            "shard-05.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-05.tar.gz",
            "shard-06.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-06.tar.gz",
            "shard-07.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-07.tar.gz",
            "shard-08.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-08.tar.gz",
            "shard-09.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-09.tar.gz",
            "shard-10.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-10.tar.gz",
            "shard-11.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-11.tar.gz",
            "shard-12.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-12.tar.gz",
            "shard-13.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-13.tar.gz",
            "shard-14.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-14.tar.gz",
            "shard-15.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-15.tar.gz",
            "shard-16.tar.gz": "https://data.vision.ee.ethz.ch/yawli/shard-16.tar.gz",
        },
        "val": {
            "val1.tar.gz": "https://data.vision.ee.ethz.ch/yawli/val1.tar.gz",
        },
    }

    # for integrity of downloaded data
    checksums = {
        "train": "a83bdb97076d617e4965913195cc84d1",
        "val": "972ba478c530b76eb9404b038597f65f",
    }

    def __init__(
        self,
        root: str,
        mode: str = "train",
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.mode = mode
        self.transform = transform

        if self.mode == "train":
            # train_folder_names = ['0001000', ..., '0085000']
            train_folder_names = [str(i * 1000).zfill(7) for i in range(1, 86)]
            self.img_dirs = [
                os.path.join(self.root, folder) for folder in train_folder_names
            ]
        elif self.mode == "val":
            self.img_dirs = [os.path.join(self.root, "val1", "HR", "val")]
        else:
            raise ValueError(
                f"Expected `train` or `val` values for `mode` argument, instead got `{self.mode}`"
            )

        # download a split of the dataset, we check first that this split isn't already downloaded
        if download:
            raise ValueError(
                f"""The official site hosting the dataset is unavailable : https://data.vision.ee.ethz.ch/yawli/.\n
                    Thus the download argument isn't working for now."""
            )
            if not os.path.isdir(self.root):
                os.makedirs(self.root)
            # if a folder image exists, we stop the download
            if any([os.path.exists(img_dir) for img_dir in self.img_dirs]):
                raise ValueError(
                    f"The {self.mode} folders already exists, thus the download is aborted. Please set `download=False` OR remove `{self.img_dirs}`."
                )

            for filename, url in self.archive_urls[self.mode].items():
                # download tar file from the Internet and save it locally
                download_archive(
                    url=url,
                    save_path=os.path.join(self.root, filename),
                )
                # extract local tar file
                extract_tarball(os.path.join(self.root, filename), self.root)

            if self.verify_split_dataset_integrity():
                print("Dataset has been successfully downloaded.")
            else:
                raise ValueError("There is an issue with the data downloaded.")

        self.img_paths = []
        for img_dir in self.img_dirs:
            try:
                self.img_paths.extend(
                    [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "Data folder doesn't exist, please set `download=True`"
                )
        self.img_paths = sorted(self.img_paths)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Any:
        img_path = self.img_paths[idx]
        # PIL Image
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def verify_split_dataset_integrity(self) -> bool:
        """Verify the integrity and existence of the specified dataset split.

        The expected structure of the dataset directory is as follows: ::

            self.root --- 0001000 --- 0000001.png
                       |           |
                       |           -- 0001000.png
                       |  ...
                       |
                       -- 0085000 --- 0084001.png
                       |           |
                       |           -- 0084991.png
                       |
                       -- val1 --- HR --- val --- 0000001.png
                       |                       |
                       |                       -- 0000250.png
                       -- xxx
        """
        root_dir_exist = os.path.isdir(self.root)
        if not root_dir_exist:
            return False

        md5_folders = hashlib.md5()
        for img_dir in self.img_dirs:
            md5_folders.update(calculate_md5_for_folder(img_dir).encode())
        return md5_folders.hexdigest() == self.checksums[self.mode]
