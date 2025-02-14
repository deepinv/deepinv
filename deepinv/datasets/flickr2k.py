from typing import Any, Callable
import os

from PIL import Image
import torch

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_zipfile,
)


class Flickr2kHR(torch.utils.data.Dataset):
    """Dataset for `Flickr2K <https://github.com/limbee/NTIRE2017>`_.

    **Raw data file structure:** ::

        self.root --- Flickr2K --- 000001.png
                   |            |
                   |            -- 002650.png
                   |
                   -- Flickr2K.zip

    | Partial raw dataset source (only HR images) : https://huggingface.co/datasets/goodfellowliu/Flickr2K/resolve/main/Flickr2K.zip
    | Full raw dataset source (HR and LR images) : https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional)  A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet ::

            from deepinv.datasets import Flickr2kHR
            root = "/path/to/dataset/Flickr2K"
            dataset = Flickr2kHR(root=root, download=True)  # download raw data at root and load dataset
            print(dataset.check_dataset_exists())           # check that raw data has been downloaded correctly
            print(len(dataset))                             # check that we have 100 images

    """

    archive_urls = {
        "Flickr2K.zip": "https://huggingface.co/datasets/goodfellowliu/Flickr2K/resolve/main/Flickr2K.zip",
    }

    # for integrity of downloaded data
    checksums = {
        "Flickr2K": "21fc3b64443fba44d6f0ad8a8c171b1e",
    }

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(self.root, "Flickr2K")

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
                    # download zip file from the Internet and save it locally
                    download_archive(
                        url=url,
                        save_path=os.path.join(self.root, filename),
                    )
                    # extract local zip file
                    extract_zipfile(os.path.join(self.root, filename), self.root)

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

            self.root --- Flickr2K --- 000001.png
                       |            |
                       |            -- 002650.png
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
