from typing import Any, Callable
import os

from PIL import Image
import torch

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_tarball,
)


class Urban100HR(torch.utils.data.Dataset):
    """Dataset for `Urban100 <https://paperswithcode.com/dataset/urban100>`_.

    The Urban100 dataset contains 100 images of urban scenes.
    It is commonly used as a test set to evaluate the performance of super-resolution models.


    **Raw data file structure:** ::

        self.root --- Urban100_HR --- img_001.png
                   |               |
                   |               -- img_100.png
                   |
                   -- Urban100_HR.tar.gz

    This dataset wrapper gives access to the 100 high resolution images in the Urban100_HR folder.
    Raw dataset source : https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional)  A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet

        >>> import shutil
        >>> from deepinv.datasets import Urban100HR
        >>> dataset = Urban100HR(root="Urban100", download=True)  # download raw data at root and load dataset
        Dataset has been successfully downloaded.
        >>> print(dataset.check_dataset_exists())                      # check that raw data has been downloaded correctly
        True
        >>> print(len(dataset))                                        # check that we have 100 images
        100
        >>> shutil.rmtree("Urban100")                             # remove raw data from disk

    """

    archive_urls = {
        "Urban100_HR.tar.gz": "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz",
    }

    # for integrity of downloaded data
    checksums = {
        "Urban100_HR": "6e0640850d436a359e0a9baf5eabd27b",
    }

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(self.root, "Urban100_HR")

        # download dataset, we check first that dataset isn't already downloaded
        if not self.check_dataset_exists():
            if download:
                if not os.path.isdir(self.root):
                    os.makedirs(self.root)
                if os.path.exists(self.img_dir):
                    raise ValueError(
                        f"The image folder already exists, thus the download is aborted. Please set `download=False` OR remove `{self.img_dir}`."
                    )

                for filename, url in self.archive_urls.items():
                    # download tar file from the Internet and save it locally
                    download_archive(
                        url=url,
                        save_path=os.path.join(self.root, filename),
                    )
                    # extract local tar file
                    extract_tarball(os.path.join(self.root, filename), self.root)

                    if self.check_dataset_exists():
                        print("Dataset has been successfully downloaded.")
                    else:
                        raise ValueError("There is an issue with the data downloaded.")
            # stop the execution since the dataset is not available and we didn't download it
            else:
                raise RuntimeError(
                    f"Dataset not found at `{self.root}`. Please set `root` correctly (currently `root={self.root}`) OR set `download=True` (currently `download={download}`)."
                )

        self.img_list = sorted(os.listdir(self.img_dir))

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Any:
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        # PIL Image
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def check_dataset_exists(self) -> bool:
        """Verify that the image folders exist and contain all the images.

        ``self.root`` should have the following structure: ::

            self.root --- Urban100_HR --- img_001.png
                       |               |
                       |               -- img_100.png
                       |
                       -- xxx
        """
        data_dir_exist = os.path.isdir(self.root)
        if not data_dir_exist:
            return False
        return all(
            calculate_md5_for_folder(os.path.join(self.root, folder_name)) == checksum
            for folder_name, checksum in self.checksums.items()
        )
