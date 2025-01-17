from typing import Union, Callable
from pathlib import Path
import os

from natsort import natsorted
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose

from deepinv.datasets.utils import (
    download_archive,
    extract_zipfile,
    calculate_md5_for_folder,
    loadmat,
)
from deepinv.utils.demo import get_image_url
from deepinv.utils.tensorlist import TensorList


class NBUDataset(Dataset):
    """NBU remote sensing multispectral satellite imagery dataset.

    Returns ``Cx256x256`` multispectral (MS) satellite images of urban scenes from 6 different satellites.
    with ``C=4`` for ``"gaofen-1"`` and ``C=8`` for the rest.

    For pan-sharpening problems, you can return pan-sharpening measurements by using ``return_pan=True``,
    outputting a :class:`deepinv.utils.TensorList` of ``(MS, PAN)`` where ``PAN`` are 1024x1024 panchromatic images.

    This dataset was compiled in `A Large-Scale Benchmark Data Set for Evaluating Pansharpening Performance <https://ieeexplore.ieee.org/document/9082183>`_
    and downloaded from `this drive <https://github.com/Lihui-Chen/Awesome-Pansharpening?tab=readme-ov-file#datasets>`_.
    We perform no other processing other than to take the "Urban" subset and provide each satellite's data separately, which you can choose using the ``satellite`` argument:

    - ``"gaofen-1"``: 5 images
    - ``"ikonos"``: 60 images
    - ``"quickbird"``: 150 images
    - ``"worldview-2"``: 150 images
    - ``"worldview-3"``: 55 images
    - ``"worldview-4"``: 90 images

    .. note::

        Returns images as :class:`torch.Tensor` normalised to 0-1 over the whole dataset.

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet ::

            from deepinv.datasets import NBUDataset
            dataset = NBUDataset(
                root_dir=".",            # root directory
                satellite="worldview-2", # choose satellite
                download=True,           # download dataset
                return_pan=True          # return panchromatic image too as pair (MS, PAN)
            )
            print(dataset.check_dataset_exists())
            print(len(dataset))

    :param str, pathlib.Path root_dir: NBU dataset root directory
    :param str satellite: satellite name, choose from the options above, defaults to "gaofen-1".
    :param bool return_pan: if ``True``, return panchromatic images as TensorList of (MS, PAN), if ``False``, just return multispectral images.
    :param Callable transform_ms: optional transform for multispectral images
    :param Callable transform_pan: optional transform for panchromatic images
    :param bool download: whether to download dataset
    """

    satellites = {
        "ikonos": "cf6fdb64ca5fbbf7050b8e27b2f9399d",
        "gaofen-1": "ea1525b7bd5342f0177d898e3c44bb51",
        "quickbird": "47163aec0a0be2c98ee267166d8aa5d3",
        "worldview-2": "11310cee5a8dd5ee0dc3b79b6b3c3203",
        "worldview-3": "85e5f7027fb7bde8592284b060fe145e",
        "worldview-4": "3a3ade874e0095978648132501edfc01",
    }

    def __init__(
        self,
        root_dir: Union[str, Path],
        satellite: str = "gaofen-1",
        return_pan: bool = False,
        transform_ms: Callable = None,
        transform_pan: Callable = None,
        download: bool = False,
    ):
        if satellite not in self.satellites:
            raise ValueError(
                'satellite must be "ikonos", "gaofen-1", "quickbird", "worldview-2", "worldview-3", or "worldview-4".'
            )

        self.data_dir = Path(root_dir) / "nbu" / satellite
        self.normalise = lambda x: (
            x / (1023 if satellite == "gaofen-1" else 2047)
        ).astype(np.float32)
        self.transform_ms = transform_ms
        self.transform_pan = transform_pan
        self.return_pan = return_pan

        if not self.check_dataset_exists():
            if download:
                dl_file = str(self.data_dir) + ".zip"
                print(f"Downloading {dl_file}")
                download_archive(get_image_url(f"nbu_{satellite}.zip"), dl_file)
                extract_zipfile(dl_file, self.data_dir.parent)
                os.remove(dl_file)

                if self.check_dataset_exists():
                    print("Dataset has been successfully downloaded.")
                else:
                    raise ValueError("There is an issue with the data downloaded.")
            else:
                raise FileNotFoundError(
                    "Local dataset not downloaded or root set incorrectly. Download by setting download=True."
                )

        self.ms_paths = natsorted(self.data_dir.glob("MS_256/*.mat"))
        self.pan_paths = natsorted(self.data_dir.glob("PAN_1024/*.mat"))
        assert len(self.ms_paths) == len(self.pan_paths), "Image dataset incomplete."
        self.image_paths = list(zip(self.ms_paths, self.pan_paths))
        for _ms, _pan in self.image_paths:
            assert _ms.name == _pan.name, "MS and PAN filenames do not match."

    def check_dataset_exists(self):
        """Verify that the image folders exist and contain all the images.

        ``root_dir`` should have the following structure: ::

            root_dir --- nbu --- <satellite> --- 1.mat
                      |       |               |
                      |       |               -- x.mat
                      |       -- <satellite>
                      -- xxx
        """
        return (
            os.path.isdir(self.data_dir)
            and len(list(self.data_dir.glob("MS_256/*.mat"))) > 0
            and calculate_md5_for_folder(str(self.data_dir / "MS_256"))
            == self.satellites[self.data_dir.stem]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load satellite image and convert to tensor.

        :param int idx: image index
        :return: torch.Tensor: normalised image to the range [0,1]
        """
        paths = self.image_paths[idx]
        ms, pan = loadmat(paths[0])["imgMS"], loadmat(paths[1])["imgPAN"]

        transform_ms = Compose(
            [self.normalise, ToTensor()]
            + ([self.transform_ms] if self.transform_ms is not None else [])
        )
        transform_pan = Compose(
            [self.normalise, ToTensor()]
            + ([self.transform_pan] if self.transform_pan is not None else [])
        )

        ms = transform_ms(ms)
        pan = transform_pan(pan)

        return TensorList([ms, pan]) if self.return_pan else ms
