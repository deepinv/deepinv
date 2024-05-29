import os

from PIL import Image
import torch

from .utils import calculate_md5_for_folder, download_zipfile, extract_zipfile


class DIV2K(torch.utils.data.Dataset):
    """Dataset for `DIV2K Image Super-Resolution Challenge <https://data.vision.ee.ethz.ch/cvl/DIV2K>`_.

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param str mode: Select a subset of the dataset between 'train' or 'val'. Default at 'train'.
    :param bool download: If True, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param callable, optional transform: A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
    """

    # https://data.vision.ee.ethz.ch/cvl/DIV2K/
    zipfile_urls = {
        "DIV2K_train_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "DIV2K_valid_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    }

    # for integrity of downloaded data
    checksums = {
        "DIV2K_train_HR": "f9de9c251af455c1021017e61713a48b",
        "DIV2K_valid_HR": "542325e500b0a474c7ad18bae922da72",
    }

    def __init__(
        self, root: str, mode: str = "train", download: bool = False, transform=None
    ) -> None:
        super().__init__()

        self.root = root
        self.mode = mode
        self.transform = transform
        train_dir_path = os.path.join(self.root, "DIV2K_train_HR")
        valid_dir_path = os.path.join(self.root, "DIV2K_valid_HR")

        # verify that dataset is available at self.root and use it
        if self._check_dataset_exists():
            print(
                f"""
            Dataset is available at `{self.root}`.
            `download` flag is not taken into account.
            """
            )
        # otherwise we try to download the whole dataset
        elif download:
            if not os.path.isdir(self.root):
                os.makedirs(self.root)
            if os.path.exists(train_dir_path):
                raise ValueError(
                    f"""
                The train folder already exists,
                thus the download is aborted.
                Please set `download=False`
                OR remove `{train_dir_path}`."""
                )

            if os.path.exists(valid_dir_path):
                raise ValueError(
                    f"""
                The val folder already exists,
                thus the download is aborted.
                Please set `download=False`
                OR remove `{valid_dir_path}`."""
                )

            for filename, url in self.zipfile_urls.items():
                # download zipfile from the Internet and save it locally
                download_zipfile(
                    url=self.zipfile_urls[filename],
                    save_path=os.path.join(self.root, filename),
                )
                # extract local zipfile
                extract_zipfile(os.path.join(self.root, filename), self.root)
        # stop the execution since the dataset is not available and we didn't download it
        else:
            raise RuntimeError(
                f"""
            Dataset not found at `{self.root}`.
            Please set `root` correctly (currently `root={self.root}`),
            AND check that `{train_dir_path}` contains ONLY the following files 0001.png, ..., 0800.png
                           `{valid_dir_path}` contains ONLY the followinf files 0801.png, ..., 0900.png
            OR set `download=True` (currently `download={download}`).
            """
            )

        if self.mode == "train":
            self.img_dir = train_dir_path
        elif self.mode == "val":
            self.img_dir = valid_dir_path
        else:
            raise ValueError(
                f"Expected `train` or `val` values for `mode` argument, instead got `{self.mode}`"
            )

        self.img_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        # PIL Image
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def _check_dataset_exists(self):
        """Verify that the train and val folders exist and contain all images.

        We verify that `self.root` has the following structure:
            self.root --- DIV2K_train_HR --- 0001.png
                       |                  |
                       |                  -- 0800.png
                       |
                       -- DIV2K_valid_HR --- 0801.png
                       |                  |
                       |                  -- 0900.png
                       -- xxx
        """
        data_dir_exist = os.path.isdir(self.root)
        if not data_dir_exist:
            return False
        return all(
            calculate_md5_for_folder(os.path.join(self.root, split)) == checksum
            for split, checksum in self.checksums.items()
        )
