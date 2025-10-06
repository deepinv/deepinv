import os
import os.path
from PIL import Image
from pathlib import Path
from deepinv.datasets.utils import calculate_md5, download_archive, extract_zipfile
from deepinv.datasets.base import ImageDataset
from natsort import natsorted


class BSDS500(ImageDataset):
    """Dataset for `BSDS500 <https://github.com/BIDS/BSDS500>`_.

    BSDS500 dataset for image restoration benchmarks. BSDS stands for The Berkeley Segmentation Dataset and Benchmark from :footcite:t:`martin2001database`.
    Originally, BSDS500 was used for image segmentation. However, this dataset only loads the ground truth images.
    The dataset consists of RGB color images of size 481 x 321 or 321 x 481 and is divided into three splits:

    - "train": contains 200 training images
    - "val": contains 100 validation images
    - "test": contains 200 test images

    Despite the name, the "val" split is often used for testing (e.g., it is a superset of CBSD68), while the "train" and "test" splits are used for training.

    This dataset uses the file structure from the github repository `https://github.com/BIDS/BSDS500 <https://github.com/BIDS/BSDS500>`_
    from the institute which published the dataset.

    **Raw data file structure:** ::

            self.root --- BSDS500-master --- (all files from the github repo)


    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param bool train: If ``True``, the standard training dataset (containing the splits "train" and "test") will be loaded. If ``False``,
        the standard test set (containing the "val" split) is loaded (which is a superset of CBSD68). Default at True
    :param list of str splits: Alternatively to the `train` parameter, the precise splits used can be defined. E.g., pass `["train", "val"]`
        to load the "train" and "val" splits. None for using the splits defined by the `train` parameter. Default None.
    :param Callable transform: (optional) A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``.
    :param bool rotate: If set to ``True`` images are rotated to have all the same orientation. This can be important to use a torch dataloader.
        Default at False.
    """

    def __init__(
        self,
        root,
        download=False,
        train=True,
        splits=None,
        transform=None,
        rotate=False,
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
            download_archive(
                "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip",
                zip_path,
            )
            download_sum = calculate_md5(fpath=zip_path)
            if not download_sum == checksum:
                return ValueError(
                    "Verification of the dataset failed (unexpected md5 checksum of the downloaded zip-file)"
                )
            extract_zipfile(zip_path, self.base_path)
        if not download and not os.path.exists(zip_path):
            raise NameError(
                "Dataset does not exist. Set download=True for downloading it or choose root correctly."
            )
        image_path_train = Path(
            self.base_path, "BSDS500-master/BSDS500/data/images/train"
        )
        image_path_test = Path(
            self.base_path, "BSDS500-master/BSDS500/data/images/test"
        )
        image_path_val = Path(self.base_path, "BSDS500-master/BSDS500/data/images/val")

        self.file_list = []
        if "train" in splits:
            self.file_list.extend(natsorted(image_path_train.glob("*.jpg")))
        if "test" in splits:
            self.file_list.extend(natsorted(image_path_test.glob("*.jpg")))
        if "val" in splits:
            self.file_list.extend(natsorted(image_path_val.glob("*.jpg")))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx]).convert("RGB")
        if self.rotate:
            width, height = img.size
            if width > height:
                img = img.transpose(Image.ROTATE_90)
        if self.transforms is not None:
            img = self.transforms(img)
        return img
