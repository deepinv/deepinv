from typing import Any, Callable
import os

from PIL import Image
import torch

from deepinv.datasets.utils import calculate_md5_for_folder, download_zipfile, extract_zipfile


class Urban100HR(torch.utils.data.Dataset):
    """Dataset for `Urban100 <https://paperswithcode.com/dataset/urban100>`_.

    The Urban100 dataset contains 100 images of urban scenes.
    It is commonly used as a test set to evaluate the performance of super-resolution models.


    **Raw data file structure:** ::

        self.root --- image_SRF_2 --- img_001_SRF_2_A+.png
                   |               |
                   |               -- img_100_SRF_2_SRCNN.png
                   |
                   -- image_SRF_4 --- img_001_SRF_4_A+.png
                   |               |
                   |               -- img_100_SRF_4_SRCNN.png
                   -- readme.txt
                   -- source_selected.xlsx
                   -- Urban100_SR.zip

    This dataset wrapper gives access to the 100 high resolution images in the `image_SRF_4` folder.
    For more information about the raw data, you can look at `readme.txt`.

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param callable, optional transform: A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instanciate dataset and download raw data from the Internet: ::
    
            root = "/path/to/dataset/Urban100"
            dataset = Urban100HR(root=root, download=True)  # will download dataset at root
            dataset.check_dataset_exists()                  # check that raw data has been downloaded correctly
            print(len(dataset))
            assert len(dataset) == 100                      # check that we have 100 images

    """

    zipfile_urls = {
        "Urban100_SR.zip": "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip",
    }

    # for integrity of downloaded data
    checksums = {
        "image_SRF_2": "7a69080e004abff22afea2520f7d7e83",
        "image_SRF_4": "7c4479537ef7bf42270cf663205a136b",
    }

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(self.root, "image_SRF_4")

        # download dataset, we check first that dataset isn't already downloaded
        if not self.check_dataset_exists():
            if download:
                if not os.path.isdir(self.root):
                    os.makedirs(self.root)
                if os.path.exists(self.img_dir):
                    raise ValueError(
                    f"""The image folder already exists,
                    thus the download is aborted.
                    Please set `download=False`
                    OR remove `{self.img_dir}`."""
                    )

                for filename, url in self.zipfile_urls.items():
                    # download zipfile from the Internet and save it locally
                    download_zipfile(
                        url=url,
                        save_path=os.path.join(self.root, filename),
                    )
                    # extract local zipfile
                    extract_zipfile(os.path.join(self.root, filename), self.root)

                    if self.check_dataset_exists():
                        print("Dataset has been successfully downloaded.")
            # stop the execution since the dataset is not available and we didn't download it
            else:
                raise RuntimeError(
                f"""Dataset not found at `{self.root}`.
                Please set `root` correctly (currently `root={self.root}`)
                OR set `download=True` (currently `download={download}`)."""
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

        `self.root` should have the following structure: ::

            self.root --- image_SRF_2 --- img_001_SRF_2_A+.png
                       |               |
                       |               -- img_100_SRF_2_SRCNN.png
                       |
                       -- image_SRF_4 --- img_001_SRF_4_A+.png
                       |               |
                       |               -- img_100_SRF_4_SRCNN.png
                       -- xxx
        """
        data_dir_exist = os.path.isdir(self.root)
        if not data_dir_exist:
            return False
        return all(
            calculate_md5_for_folder(os.path.join(self.root, folder_name)) == checksum
            for folder_name, checksum in self.checksums.items()
        )
