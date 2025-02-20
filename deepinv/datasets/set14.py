from typing import Any, Callable
import os

from PIL import Image
import torch

from deepinv.datasets.utils import (
    calculate_md5_for_folder,
    download_archive,
    extract_zipfile,
)


class Set14HR(torch.utils.data.Dataset):
    """Dataset for `Set14 <https://paperswithcode.com/dataset/set14>`_.

    The Set14 dataset is a dataset consisting of 14 images commonly used for testing performance of image reconstruction algorithms.
    Images have sizes ranging from 276×276 to 512×768 pixels.

    **Raw data file structure:** ::

        self.root --- Set14 --- image_SRF_2 --- img_001_SRF_2_bicubic.png
                   |         |               |
                   |         |               -- img_014_SRF_2_SRCNN.png
                   |         |
                   |         -- image_SRF_3 --- ...
                   |         -- image_SRF_4 --- ...
                   |
                   -- Set14_SR.zip

    This dataset wrapper gives access to the 14 high resolution images in the `image_SRF_4` folder.
    Raw dataset source : https://github.com/jbhuang0604/SelfExSR

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

    archive_urls = {
        "Set14_SR.zip": "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip",
    }

    # for integrity of downloaded data
    checksums = {
        "image_SRF_2": "f51503d396f9419192a8075c814bcee3",
        "image_SRF_3": "05130ee0f318dde02064d98b1e2019bc",
        "image_SRF_4": "2b1bcbde607e6188ddfc526b252c0e1a",
    }

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(self.root, "Set14", "image_SRF_4")

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

        self.img_list = sorted(
            [file for file in os.listdir(self.img_dir) if file.endswith("HR.png")]
        )

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

            self.root --- Set14 --- image_SRF_2 --- img_001_SRF_2_bicubic.png
                       |         |               |
                       |         |               -- img_014_SRF_2_SRCNN.png
                       |         |
                       |         -- image_SRF_3 --- ...
                       |         -- image_SRF_4 --- ...
                       |         -- xxx
                       -- xxx
        """
        data_dir_exist = os.path.isdir(os.path.join(self.root, "Set14"))
        if not data_dir_exist:
            return False
        return all(
            calculate_md5_for_folder(os.path.join(self.root, "Set14", folder_name))
            == checksum
            for folder_name, checksum in self.checksums.items()
        )
