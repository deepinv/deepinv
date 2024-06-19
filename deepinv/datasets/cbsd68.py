from typing import Any, Callable

import datasets
import torch

from deepinv.datasets.utils import calculate_md5_for_folder


class CBSD68(torch.utils.data.Dataset):
    """Dataset for `CBSBD68 <https://paperswithcode.com/dataset/cbsd68>`_.

    Color BSD68 dataset for image denoising benchmarks is part of The Berkeley Segmentation Dataset and Benchmark. 
    It is used for measuring image denoising algorithms performance. It contains 68 images.

    
    **Raw data file structure:** ::

            self.root --- data-00000-of-00001.arrow
                       -- dataset_info.json
                       -- state.json

    This dataset wraps the huggingface version of the dataset.
    HF source : https://huggingface.co/datasets/deepinv/CBSD68
                       
    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param callable, optional transform: A function/transform that takes in a PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
    """

    # for integrity of downloaded data
    checksum = "71e89aded7583f4c6b4e8aad5ccb51e5"

    def __init__(
        self,
        root: str,
        download: bool = False,
        transform: Callable = None,
    ) -> None:
        self.root = root
        self.transform = transform

        # download dataset, we check first that dataset isn't already downloaded
        if not self.check_dataset_exists():
            if download:
                hf_dataset = datasets.load_dataset("deepinv/CBSD68", split="train")
                hf_dataset.save_to_disk(self.root)

                if self.check_dataset_exists():
                    print("Dataset has been successfully downloaded.")
            # stop the execution since the dataset is not available and we didn't download it
            else:
                raise RuntimeError(
                    f"Dataset not found at `{self.root}`. Please set `root` correctly (currently `root={self.root}`) OR set `download=True` (currently `download={download}`)."
                )
            
        self.hf_dataset = datasets.load_from_disk(self.root)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Any:
        # PIL Image
        img = self.hf_dataset[idx]['png']

        if self.transform is not None:
            img = self.transform(img)
        return img


    def check_dataset_exists(self) -> bool:
        """Verify that the HuggingFace dataset folder exists and contains all the files.

        `self.root` should have the following structure: ::

            self.root --- data-00000-of-00001.arrow
                       -- dataset_info.json
                       -- state.json
        """
        return calculate_md5_for_folder(self.root) == self.checksum
