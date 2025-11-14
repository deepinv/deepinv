from typing import Callable
from types import MappingProxyType
import os

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_tarball,
)
from deepinv.datasets.base import ImageFolder


class Set14HR(ImageFolder):
    """Dataset for `Set14 <https://paperswithcode.com/dataset/set14>`_.

    The Set14 dataset :footcite:p:`huang2015single` is a dataset consisting of 14 images commonly used for testing performance of image reconstruction algorithms.
    Images have sizes ranging from 276×276 to 512×768 pixels.

    **Raw data file structure:** ::

        self.root --- Set14_HR.tar.gz
                |
                --- Set14_HR --- baboon.png
                |             |
                |             --- butterfly.png
                |             --- face.png
                |             --- ...
                |
                --- xxx

    This dataset wrapper gives access to the 14 high resolution images in the `Set14_HR` folder.
    Raw dataset source : https://huggingface.co/datasets/eugenesiow/Set14

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional)  A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet ::

            import shutil
            from deepinv.datasets import Set14HR
            dataset = Set14HR(root="Set14", download=True)  # download raw data at root and load dataset
            Dataset has been successfully downloaded.
            print(dataset.check_dataset_exists())                # check that raw data has been downloaded correctly
            True
            print(len(dataset))                                  # check that we have 14 images
            14
            shutil.rmtree("Set14")                          # remove raw data from disk

    """

    _archive_urls = MappingProxyType(
        {
            "Set14_HR.tar.gz": "https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz",
        }
    )
    _checksums = MappingProxyType({"Set14_HR": "3fce01c3dfe9760194e8a22f6bc032c5"})
    # for integrity of downloaded data

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.img_dir = os.path.join(self.root, "Set14_HR")

        # download dataset, we check first that dataset isn't already downloaded
        if not self.check_dataset_exists():
            if download:
                if not os.path.isdir(self.root):
                    os.makedirs(self.root)
                if os.path.exists(self.img_dir):
                    raise ValueError(
                        f"The image folder already exists, thus the download is aborted. Please set `download=False` OR remove `{self.img_dir}`."
                    )

                for filename, url in self._archive_urls.items():
                    download_archive(
                        url=url,
                        save_path=os.path.join(self.root, filename),
                    )
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

        # Initialize ImageFolder
        super().__init__(self.img_dir, transform=transform)

    def check_dataset_exists(self) -> bool:
        """Verify that the image folders exist and contain all the images.

        ``self.root`` should have the following structure: ::

            self.root --- Set14_HR --- baboon.png
                    |             |
                    |             --- butterfly.png
                    |             --- face.png
                    |             --- ...
                    |
                    --- xxx
        """
        data_dir_exist = os.path.isdir(os.path.join(self.root, "Set14_HR"))
        if not data_dir_exist:
            return False
        return all(
            calculate_md5_for_folder(os.path.join(self.root, folder_name)) == checksum
            for folder_name, checksum in self._checksums.items()
        )
