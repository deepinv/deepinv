from typing import Callable
from types import MappingProxyType
import os

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_tarball,
)
from deepinv.datasets.base import ImageFolder
from .utils import resolve_root


class BSD100HR(ImageFolder):
    """Dataset for `BSD100 <https://paperswithcode.com/dataset/bsd100>`_.

    The BSD100 dataset :footcite:p:`martin2001database` is a dataset consisting of 100 images commonly used for testing performance of image reconstruction algorithms.
    Images have sizes ranging from 240×160 to 480×320 pixels.

    **Raw data file structure:** ::

        self.root --- BSD100_HR.tar.gz
                |
                --- BSD100_HR --- 3096.png
                |               |
                |               --- 8023.png
                |               --- 12084.png
                |               --- ...
                |
                --- xxx

    Raw dataset source : https://huggingface.co/datasets/eugenesiow/BSD100

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional)  A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
    :param bool verbose: Print a message if the dataset has been correctly downloaded. Default ``True``.
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).

    """

    _archive_urls = MappingProxyType(
        {
            "BSD100_HR.tar.gz": "https://huggingface.co/datasets/eugenesiow/BSD100/resolve/main/data/BSD100_HR.tar.gz",
        }
    )
    _checksums = MappingProxyType({"BSD100_HR": "2288d442262c1c26343fda3b36f05b03"})
    # for integrity of downloaded data

    def __init__(
        self,
        root: str = None,
        download: bool = False,
        transform: Callable = None,
        verbose: bool = True,
        use_dict_output: bool = False,
    ) -> None:
        self.root = resolve_root(root, "BSD100")
        self.img_dir = os.path.join(self.root, "BSD100_HR")

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

                if self.check_dataset_exists() and verbose:
                    print("Dataset has been successfully downloaded.")
                else:
                    raise ValueError("There is an issue with the data downloaded.")
            # stop the execution since the dataset is not available and we didn't download it
            else:
                raise RuntimeError(
                    f"Dataset not found at `{self.root}`. Please set `root` correctly (currently `root={self.root}`) OR set `download=True` (currently `download={download}`)."
                )

        # Initialize ImageFolder
        super().__init__(
            self.img_dir, transform=transform, use_dict_output=use_dict_output
        )

    def check_dataset_exists(self) -> bool:
        """Verify that the image folders exist and contain all the images.

        ``self.root`` should have the following structure: ::

            self.root --- BSD100_HR --- 3096.png
                    |               |
                    |               --- 8023.png
                    |               --- 12084.png
                    |               --- ...
                    |
                    --- xxx
        """
        data_dir_exist = os.path.isdir(os.path.join(self.root, "BSD100_HR"))
        if not data_dir_exist:
            return False
        return all(
            calculate_md5_for_folder(os.path.join(self.root, folder_name)) == checksum
            for folder_name, checksum in self._checksums.items()
        )
