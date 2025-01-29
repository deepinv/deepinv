from typing import Any, Callable
import os

import torch

from deepinv.datasets.utils import calculate_md5

error_import = None
try:
    from datasets import load_dataset as load_dataset_hf, load_from_disk
except:
    error_import = ImportError(
        "datasets is not available. Please install the datasets package with `pip install datasets`."
    )


class CBSD68(torch.utils.data.Dataset):
    """Dataset for `CBSBD68 <https://paperswithcode.com/dataset/cbsd68>`_.

    Color BSD68 dataset for image restoration benchmarks is part of The Berkeley Segmentation Dataset and Benchmark.
    It is used for measuring image restoration algorithms performance. It contains 68 images.


    **Raw data file structure:** ::

            self.root --- data-00000-of-00001.arrow
                       -- dataset_info.json
                       -- state.json

    This dataset wraps the huggingface version of the dataset.
    HF source : https://huggingface.co/datasets/deepinv/CBSD68

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform: (optional) A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet

        >>> import shutil
        >>> from deepinv.datasets import CBSD68
        >>> dataset = CBSD68(root="CBSB68", download=True)  # download raw data at root and load dataset
        Dataset has been successfully downloaded.
        >>> print(dataset.check_dataset_exists())                # check that raw data has been downloaded correctly
        True
        >>> print(len(dataset))                                  # check that we have 68 images
        68
        >>> shutil.rmtree("CBSB68")                         # remove raw data from disk

    """

    # for integrity of downloaded data
    checksum = "18e128fbf5bb99ea7fca35f59683ea39"

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        if error_import is not None and isinstance(error_import, ImportError):
            raise error_import

        self.root = root
        self.transform = transform

        # download dataset, we check first that dataset isn't already downloaded
        if not self.check_dataset_exists():
            if download:
                # source : https://github.com/huggingface/datasets/issues/6703
                # load_dataset : download from Internet, raw data formats like CSV are processed into Arrow format, then saved in a cache dir
                hf_dataset = load_dataset_hf("deepinv/CBSD68", split="train")

                # '__url__' column contains absolute paths to raw data in the cache dir which is unnecessary when saving the dataset
                if "__url__" in hf_dataset.column_names:
                    # Remove the '__url__' column
                    hf_dataset = hf_dataset.remove_columns("__url__")
                hf_dataset.save_to_disk(self.root)

                if self.check_dataset_exists():
                    print("Dataset has been successfully downloaded.")
                else:
                    raise ValueError("There is an issue with the data downloaded.")
            # stop the execution since the dataset is not available and we didn't download it
            else:
                raise RuntimeError(
                    f"Dataset not found at `{self.root}`. Please set `root` correctly (currently `root={self.root}`) OR set `download=True` (currently `download={download}`)."
                )

        self.hf_dataset = load_from_disk(self.root)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Any:
        # PIL Image
        img = self.hf_dataset[idx]["png"]

        if self.transform is not None:
            img = self.transform(img)
        return img

    def check_dataset_exists(self) -> bool:
        """Verify that the HuggingFace dataset folder exists and contains the raw data file.

        ``self.root`` should have the following structure: ::

            self.root --- data-00000-of-00001.arrow
                       -- xxx
                       -- xxx

        This is a soft verification as we don't check all the files in the folder.
        """
        raw_data_fpath = os.path.join(self.root, "data-00000-of-00001.arrow")
        if not os.path.exists(raw_data_fpath):
            return False
        return calculate_md5(fpath=raw_data_fpath) == self.checksum
