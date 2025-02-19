from typing import Any, Callable
import os

from PIL import Image
import torch

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_zipfile,
)


class DIV2K(torch.utils.data.Dataset):
    """Dataset for `DIV2K Image Super-Resolution Challenge <https://data.vision.ee.ethz.ch/cvl/DIV2K>`_.

    Images have varying sizes with up to 2040 vertical pixels, and 2040 horizontal pixels.


    **Raw data file structure:** ::

            self.root --- DIV2K_train_HR --- 0001.png
                       |                  |
                       |                  -- 0800.png
                       |
                       -- DIV2K_valid_HR --- 0801.png
                       |                  |
                       |                  -- 0900.png
                       -- DIV2K_train_HR.zip
                       -- DIV2K_valid_HR.zip

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param str mode: Select a split of the dataset between 'train' or 'val'. Default at 'train'.
    :param bool download: If True, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional)  A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet

        >>> import shutil
        >>> from deepinv.datasets import DIV2K
        >>> dataset = DIV2K(root="DIV2K", mode="val", download=True)  # download raw data at root and load dataset
        Dataset has been successfully downloaded.
        >>> print(dataset.verify_split_dataset_integrity())                # check that raw data has been downloaded correctly
        True
        >>> print(len(dataset))                                            # check that we have 100 images
        100
        >>> shutil.rmtree("DIV2K")                                    # remove raw data from disk
    """

    # https://data.vision.ee.ethz.ch/cvl/DIV2K/
    archive_urls = {
        "DIV2K_train_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "DIV2K_valid_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    }

    # for integrity of downloaded data
    checksums = {
        "DIV2K_train_HR": "f9de9c251af455c1021017e61713a48b",
        "DIV2K_valid_HR": "542325e500b0a474c7ad18bae922da72",
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
            self.img_dir = os.path.join(self.root, "DIV2K_train_HR")
        elif self.mode == "val":
            self.img_dir = os.path.join(self.root, "DIV2K_valid_HR")
        else:
            raise ValueError(
                f"Expected `train` or `val` values for `mode` argument, instead got `{self.mode}`"
            )

        # download a split of the dataset, we check first that this split isn't already downloaded
        if not self.verify_split_dataset_integrity():
            if download:
                if not os.path.isdir(self.root):
                    os.makedirs(self.root)
                if os.path.exists(self.img_dir):
                    raise ValueError(
                        f"The {self.mode} folder already exists, thus the download is aborted. Please set `download=False` OR remove `{self.img_dir}`."
                    )

                zip_filename = (
                    "DIV2K_train_HR.zip"
                    if self.mode == "train"
                    else "DIV2K_valid_HR.zip"
                )
                # download zip file from the Internet and save it locally
                download_archive(
                    url=self.archive_urls[zip_filename],
                    save_path=os.path.join(self.root, zip_filename),
                )
                # extract local zip file
                extract_zipfile(os.path.join(self.root, zip_filename), self.root)

                if self.verify_split_dataset_integrity():
                    print("Dataset has been successfully downloaded.")
                else:
                    raise ValueError("There is an issue with the data downloaded.")
            # stop the execution since the split dataset is not available and we didn't download it
            else:
                raise RuntimeError(
                    f"Dataset not found at `{self.root}`. Please set `root` correctly (currently `root={self.root}`), OR set `download=True` (currently `download={download}`)."
                )

        self.img_list = os.listdir(self.img_dir)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Any:
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        # PIL Image
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def verify_split_dataset_integrity(self) -> bool:
        """Verify the integrity and existence of the specified dataset split.

        This method checks if ``DIV2K_train_HR`` or ``DIV2K_valid_HR`` folder within
        ``self.root`` exists and validates the integrity of its contents by comparing
        the MD5 checksum of the folder with the expected checksum.

        The expected structure of the dataset directory is as follows: ::

            self.root --- DIV2K_train_HR --- 0001.png
                       |                  |
                       |                  -- 0800.png
                       |
                       -- DIV2K_valid_HR --- 0801.png
                       |                  |
                       |                  -- 0900.png
                       -- xxx
        """
        root_dir_exist = os.path.isdir(self.root)
        if not root_dir_exist:
            return False
        if self.mode == "train":
            return (
                calculate_md5_for_folder(self.img_dir)
                == self.checksums["DIV2K_train_HR"]
            )
        else:
            return (
                calculate_md5_for_folder(self.img_dir)
                == self.checksums["DIV2K_valid_HR"]
            )
