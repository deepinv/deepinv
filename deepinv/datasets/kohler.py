import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image

from urllib.parse import urlparse
from os.path import basename, join
from typing import Callable, Union
from pathlib import Path


def url_basename(url: str) -> str:
    parts = urlparse(url)
    path = parts.path
    return basename(path)


class Kohler(Dataset):

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
        "GroundTruth_pngs_Image2.zip": "72ce9690c3ed1296358653396cf9576d",
        "GroundTruth_pngs_Image3.zip": "3a77c41c951367f35db52eb18496bbac",
        "GroundTruth_pngs_Image4.zip": "da440d3bf43b32bec0b7170ccd828f29",
        "BlurryImages.zip": "61ffb1434d93fca6c508976a7216d723",
    }

    # Most of the acquisitions of sharp images span exactly 199 frames but not
    # all of them and this lookup table gives each frame count for them all.
    frame_count_lookup = {
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
    @staticmethod
    def download(root: Union[str, Path]) -> None:
        for url in self.archive_urls:
            name = url_basename(url)
            checksum = self.archive_checksums[name]

            # Download the archive and verify its integrity
            download_url(url, root, filename=name, md5=checksum)

            # NOTE: Extracting individual images instead of the whole directory
            # tree makes the code for loading them clearer as it ensures that
            # all extracted images end up in the same directory.

            # Extract individual images from the archive
            path = join(root, name)
            with ZipFile(path, "r") as zipfile:
                members = zipfile.namelist()
                filter_fn = lambda member: member.endswith(".png")
                members = filter(filter_fn, members)
                zipfile.extractall(path=self.root, members=members)

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

    @staticmethod
    def select_frame(
        printout_index: int, trajectory_index: int, frame: Union[int, str]
    ) -> int:
        if isinstance(frame, int):
            frame_index = frame
        else:
            frame_count = self.get_frame_count(printout_index, trajectory_index)

            if frame == "first":
                frame_index = 1
            elif frame == "middle":
                frame_index = (frame_count + 1) // 2
            elif frame == "last":
                frame_index = frame_count
            else:
                raise ValueError(f"Unsupported frame specifier: {frame}")

        return frame_index

    @staticmethod
    def get_frame_count(printout_index: int, trajectory_index: int) -> int:
        index = (printout_index, trajectory_index)

        if index in self.frame_count_lookup:
            count = self.frame_count_lookup[index]
        else:
            count = self.frame_count_lookup["others"]

        return count
