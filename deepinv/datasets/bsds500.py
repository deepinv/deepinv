import urllib.request
from torch.utils.data import Dataset
import zipfile
import os
import os.path
from PIL import Image
import numpy as np
from deepinv.datasets.utils import calculate_md5
from deepinv.datasets.base import ImageDataset


class BSDS500(ImageDataset):
    def __init__(
        self,
        root,
        download=False,
        train=True,
        transform=None,
        rotate=False,
        splits=None,
    ):
        checksum = "7bfe17302a219367694200a61ce8256c"
        if splits is None:
            if train:
                splits = ["train", "test"]
            else:
                splits = ["val"]
        self.base_path = root
        self.rotate = rotate
        self.transforms = transform
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        zip_path = os.path.join(self.base_path, "download.zip")
        if download and not os.path.exists(zip_path):
            urllib.request.urlretrieve(
                "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
                zip_path,
            )
            download_sum = calculate_md5(fpath=zip_path)
            if not download_sum == checksum:
                return ValueError(
                    "Verification of the dataset failed (unexpected md5 checksum of the downloaded zip-file)"
                )
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.base_path)
        if not download and not os.path.exists(zip_path):
            raise NameError(
                "Dataset does not exist. Set download=True for downloading it or choose root correctly."
            )
        image_path_train = os.path.join(
            self.base_path, "BSDS500-master/BSDS500/data/images/train"
        )
        image_path_test = os.path.join(
            self.base_path, "BSDS500-master/BSDS500/data/images/test"
        )
        image_path_val = os.path.join(
            self.base_path, "BSDS500-master/BSDS500/data/images/val"
        )
        self.file_list = []
        if "train" in splits:
            file_list = os.listdir(image_path_train)
            self.file_list = self.file_list + [
                os.path.join(image_path_train, f)
                for f in file_list
                if f.endswith("jpg")
            ]
        if "test" in splits:
            file_list = os.listdir(image_path_test)
            self.file_list = self.file_list + [
                os.path.join(image_path_test, f) for f in file_list if f.endswith("jpg")
            ]
        if "val" in splits:
            file_list = os.listdir(image_path_val)
            self.file_list = self.file_list + [
                os.path.join(image_path_val, f) for f in file_list if f.endswith("jpg")
            ]

        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, IDX):
        img = Image.open(self.file_list[IDX]).convert("RGB")
        img = np.array(img) / 255.0
        if self.transforms is not None:
            img = self.transforms(img)
        if self.rotate:
            if isinstance(img, (tuple, list)):
                img = [
                    i.transpose(-2, -1) if i.shape[-1] > i.shape[-2] else i for i in img
                ]
            else:
                img = img.transpose(-2, -1) if img.shape[-1] > img.shape[-2] else img
        return img
