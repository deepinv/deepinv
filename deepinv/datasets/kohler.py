import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image

from urllib.parse import urlparse
from os.path import basename, join
from typing import Callable, Union
from pathlib import Path
from zipfile import ZipFile


def url_basename(url: str) -> str:
    parts = urlparse(url)
    path = parts.path
    return basename(path)


class Kohler(Dataset):
    """Dataset for `Recording and Playback of Camera Shake: Benchmarking Blind Deconvolution with a Real-World Database <https://doi.org/10.1007/978-3-642-33786-4_3>`_.

    :param Union[str, Path] root: Root directory of the dataset.
    :param callable, optional transform: A function used to transform both the blurry shots and the sharp frames.
    :param bool download: Download the dataset.

    |sep|

    :Examples:

        Download the dataset and load one of its elements.

        >>> from deepinv.datasets import Kohler
        >>> dataset = Kohler(root="datasets/Kohler", download=True)
        >>> sharp_frame, blurry_shot = dataset.__getitem__(1, 1, frame="middle")
    """

    # The Köhler dataset is split into multiple archives available online.
    archive_urls = [
        "http://people.kyb.tuebingen.mpg.de/rolfk/BenchmarkECCV2012/GroundTruth_pngs_Image1.zip",
        "http://people.kyb.tuebingen.mpg.de/rolfk/BenchmarkECCV2012/GroundTruth_pngs_Image2.zip",
        "http://people.kyb.tuebingen.mpg.de/rolfk/BenchmarkECCV2012/GroundTruth_pngs_Image3.zip",
        "http://people.kyb.tuebingen.mpg.de/rolfk/BenchmarkECCV2012/GroundTruth_pngs_Image4.zip",
        "http://people.kyb.tuebingen.mpg.de/rolfk/BenchmarkECCV2012/BlurryImages.zip",
    ]

    # The checksums are used to verify the integrity of the downloaded
    # archives.
    archive_checksums = {
        "GroundTruth_pngs_Image1.zip": "acb90b6d9bfdb4b2370e08a5fcb80e68",
        "GroundTruth_pngs_Image2.zip": "da440d3bf43b32bec0b7170ccd828f29",
        "GroundTruth_pngs_Image3.zip": "3a77c41c951367f35db52eb18496bbac",
        "GroundTruth_pngs_Image4.zip": "72ce9690c3ed1296358653396cf9576d",
        "BlurryImages.zip": "61ffb1434d93fca6c508976a7216d723",
    }

    # Most of the acquisitions of sharp images span exactly 199 frames but not
    # all of them and this lookup table gives each frame count for them all.
    frame_count_table = {
        (2, 11): 200,
        (1, 10): 198,
        (1, 12): 198,
        (2, 10): 198,
        (3, 7): 198,
        (3, 12): 198,
        (4, 12): 198,
        "others": 199,
    }

    def __init__(
        self,
        root: Union[str, Path],
        transform: Callable = None,
        download: bool = False,
    ) -> None:
        self.transform = transform
        self.root = root

        if download:
            self.download(self.root)

    # NOTE: The archives are ultimately left in place which might be
    # inconvenient for some use cases.
    @classmethod
    def download(cls, root: Union[str, Path]) -> None:
        """Download the dataset.

        :param Union[str, Path] root: Root directory of the dataset.

        |sep|

        :Examples:

            Download the dataset.

            >>> from deepinv.datasets import Kohler
            >>> Kohler.download("datasets/Kohler")
        """
        for url in cls.archive_urls:
            archive_name = url_basename(url)
            checksum = cls.archive_checksums[archive_name]

            # Download the archive and verify its integrity
            download_url(url, root, filename=archive_name, md5=checksum)

            # NOTE: Extracting individual images instead of the whole directory
            # tree makes the code for loading them clearer as it ensures that
            # all extracted images end up in the same directory.

            # Extract individual images from the archive
            archive_path = join(root, archive_name)
            with ZipFile(archive_path, "r") as zipfile:
                infolist = zipfile.infolist()
                for info in infolist:
                    filename = basename(info.filename)
                    if filename.endswith(".png"):
                        destination_path = join(root, filename)
                        with open(destination_path, "wb") as file:
                            data = zipfile.read(info)
                            file.write(data)

    def __len__(self) -> int:
        return 48

    # While users might sometimes want to thoroughly compare their own
    # deblurred images to all the sharp frames (about 200 per blurry shot),
    # they will probably most often make the way more convenient choice of
    # comparing against a single frame per blurry shot. For this reason, the
    # function __getitem__ accepts an additional parameter for frame selection
    # and only returns the selected frame.
    def __getitem__(
        self,
        printout_index: int,
        trajectory_index: int,
        frame: Union[int, str] = "middle",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sharp frame and a blurry shot from the dataset.

        :param int printout_index: Index of the printout.
        :param int trajectory_index: Index of the trajectory.
        :param Union[int, str] frame: Frame specifier. Can be the frame number, "first", "middle", or "last".

        :return: (torch.Tensor, torch.Tensor) The sharp frame and the blurry shot.

        |sep|

        :Examples:

            >>> sharp_frame, blurry_shot = dataset.__getitem__(1, 1, frame="middle")
        """
        frame_index = self.select_frame(printout_index, trajectory_index, frame=frame)

        sharp_frame = self.open_sharp_frame(
            self.root, printout_index, trajectory_index, frame_index
        )
        blurry_shot = self.open_blurry_shot(self.root, printout_index, trajectory_index)

        if self.transform is not None:
            sharp_frame = self.transform(sharp_frame)
            blurry_shot = self.transform(blurry_shot)

        return sharp_frame, blurry_shot

    @staticmethod
    def open_sharp_frame(root, printout_index, trajectory_index, frame_index):
        path = join(
            root, f"GroundTruth{printout_index}_{trajectory_index}_{frame_index}.png"
        )
        return Image.open(path)

    @staticmethod
    def open_blurry_shot(root, printout_index, trajectory_index):
        path = join(root, f"Blurry{printout_index}_{trajectory_index}.png")
        return Image.open(path)

    @classmethod
    def select_frame(
        cls, printout_index: int, trajectory_index: int, frame: Union[int, str]
    ) -> int:
        if isinstance(frame, int):
            frame_index = frame
        else:
            frame_count = cls.get_frame_count(printout_index, trajectory_index)

            if frame == "first":
                frame_index = 1
            elif frame == "middle":
                frame_index = (frame_count + 1) // 2
            elif frame == "last":
                frame_index = frame_count
            else:
                raise ValueError(f"Unsupported frame specifier: {frame}")

        return frame_index

    @classmethod
    def get_frame_count(cls, printout_index: int, trajectory_index: int) -> int:
        index = (printout_index, trajectory_index)

        if index in cls.frame_count_table:
            count = cls.frame_count_table[index]
        else:
            count = cls.frame_count_table["others"]

        return count
