import hashlib
from typing import Callable
from types import MappingProxyType
import os

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    extract_tarball,
)
from deepinv.datasets.base import ImageFolder
import shutil


class LsdirHR(ImageFolder):
    """Dataset for `LSDIR <https://ofsoundof.github.io/lsdir-data/>`_.

    Published in :footcite:t:`li2023lsdir`.

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
                   |
                   |
                   -- val1 --- HR --- val --- 0000001.png
                   |        -- X2          |
                   |        -- X3          -- 0000250.png
                   |        -- X4

    .. warning::
        Downloading this dataset requires ``huggingface-hub``. It is gated, please request access (https://huggingface.co/ofsoundof/LSDIR) and make sure you are logged in using ``hf auth login`` (CLI) or ``from huggingface_hub import login, login()``.

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

    _archive_urls = MappingProxyType(
        {
            "train": [
                "shard-00.tar.gz",
                "shard-01.tar.gz",
                "shard-02.tar.gz",
                "shard-03.tar.gz",
                "shard-04.tar.gz",
                "shard-05.tar.gz",
                "shard-06.tar.gz",
                "shard-07.tar.gz",
                "shard-08.tar.gz",
                "shard-09.tar.gz",
                "shard-10.tar.gz",
                "shard-11.tar.gz",
                "shard-12.tar.gz",
                "shard-13.tar.gz",
                "shard-14.tar.gz",
                "shard-15.tar.gz",
                "shard-16.tar.gz",
            ],
            "val": [
                "val1.tar.gz",
            ],
        }
    )

    # for integrity of downloaded data
    _checksums = MappingProxyType(
        {
            "train": "a83bdb97076d617e4965913195cc84d1",
            "val": "972ba478c530b76eb9404b038597f65f",
        }
    )

    def __init__(
        self,
        root: str,
        mode: str = "train",
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.mode = mode

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
        if download:  # pragma: no cover
            try:
                from huggingface_hub import hf_hub_download
            except:
                raise RuntimeError(
                    "To download LsdirHR, please install huggingface-hub, request access to https://huggingface.co/ofsoundof/LSDIR (this should be instant), and authenticate yourself. For more info, see the LsdirHR's documentation."
                )
            if not os.path.isdir(self.root):
                os.makedirs(self.root)
            # if a folder image exists, we stop the download
            if any([os.path.exists(img_dir) for img_dir in self.img_dirs]):
                raise ValueError(
                    f"The {self.mode} folders already exists, thus the download is aborted. Please set `download=False` OR remove `{self.img_dirs}`."
                )

            for filename in self._archive_urls[self.mode]:
                # download tar file from HuggingFace and save it locally
                hf_hub_download(
                    repo_id="ofsoundof/LSDIR",
                    filename=filename,
                    local_dir=os.path.join(self.root),
                    cache_dir=os.path.join(self.root, ".cache/huggingface"),
                    local_dir_use_symlinks=False,
                )
                # Since LSDIR is relatively large, we want to avoid taking too much redundant disk space.
                shutil.rmtree(os.path.join(self.root, ".cache/huggingface"))

                # extract local tar file
                extract_tarball(os.path.join(self.root, filename), self.root)
                os.remove(
                    os.path.join(self.root, filename)
                )  # Since LSDIR is relatively large, we want to avoid taking too much redundant disk space.

            if self.verify_split_dataset_integrity():
                print("Dataset has been successfully downloaded.")
            else:
                raise ValueError("There is an issue with the data downloaded.")

        if not all(
            os.path.isdir(d) and os.listdir(d) for d in self.img_dirs
        ):  # pragma: no cover
            raise RuntimeError("Data folder doesn't exist, please set `download=True`")

        # Initialize ImageFolder
        if mode == "val":
            super().__init__(self.root, x_path="val1/HR/val/*.png", transform=transform)
        else:  # mode is train for sure, because of earlier check
            super().__init__(
                self.root, x_path="shard-[0-1][0-9]/**/*.png", transform=transform
            )

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
                    |
                    -- val1 --- HR --- val --- 0000001.png
                    |        -- X2          |
                    |        -- X3          -- 0000250.png
                    |
        """
        root_dir_exist = os.path.isdir(self.root)
        if not root_dir_exist:  # pragma: no cover
            return False

        md5_folders = hashlib.md5()
        for img_dir in self.img_dirs:
            md5_folders.update(calculate_md5_for_folder(img_dir).encode())
        return md5_folders.hexdigest() == self._checksums[self.mode]
