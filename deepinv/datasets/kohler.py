import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive
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
    """Dataset for `Recording and Playback of Camera Shake <https://doi.org/10.1007/978-3-642-33786-4_3>`_.

    The dataset consists of blurry shots and sharp frames, each blurry shot
    being associated with about 200 sharp frames. There are 48 blurry shots in
    total, each associated to one of 4 printouts, and to one of 12 camera
    trajectories inducing motion blur. Unlike certain deblurring datasets (e.g.
    GOPRO) where the blurry images are synthesized from sharp images, the
    blurry shots in the Köhler dataset are acquired with a real camera. It is
    the movement of the camera during exposition that causes the blur. What we
    call printouts are the 4 images that were printed out on paper and fixed to
    a screen to serve as photographed subjects — all images in the dataset show
    one of these 4 printouts.

    The ground truth images are **not** the 4 images that were printed out.
    Instead, they are the frames of videos taken in the same condition as for
    the blurry shots. The reason behind this choice is to ensure the same
    lightness for better comparison. In total, there are about 200 frames per
    video, and equivalently by blurry shot. There is a lot of redundancy
    between the frames as the camera barely moves between consecutive frames,
    for this reason the implementation allows selecting a single frame as the
    priviledged ground truth. This enables using the tooling provided by
    deepinv such as :func:`deepinv.test` and which gives approximately the same
    performance as comparing to all the frames. It is the parameter ``frames``
    that controls this behavior, when it is set to either ``"first"``,
    ``"middle"``, ``"last"``, or to a specific frame index (between 1 and 198). If
    the user wants to compare against all the frames, e.g. to reproduce the
    benchmarks of the original paper, they can do so by setting the parameter
    ``frames`` to ``"all"`` or to a list of frame indices.

    The dataset does not have a preferred ordering and this implementation
    uses lexicographic ordering on the printout index (1 to 4) and the
    trajectory index (1 to 12). The parameter ``ordering`` controls whether to
    order by printout first ``"printout_first"`` or by trajectory first
    ``"trajectory_first"``. This enables accessing the 48 items using the standard
    method ``__getitem__`` using an index between 0 and 47. The nonstandard
    method ``get_item`` allows selecting one of them by printout and trajectory
    index directly if needed.

    :param Union[int, str, list[Union[int, str]]] frames: Can be the frame number, ``"first"``, ``"middle"``, ``"last"``, or ``"all"``. If a list is provided, the method will return a list of sharp frames.
    :param str ordering: Ordering of the dataset. Can be ``"printout_first"`` or ``"trajectory_first"``.
    :param Union[str, pathlib.Path] root: Root directory of the dataset.
    :param Callable transform:: (optional)  A function used to transform both the blurry shots and the sharp frames.
    :param bool download: Download the dataset.

    |sep|

    :Examples:

        Download the dataset and load one of its elements ::

            from deepinv.datasets import Kohler
            dataset = Kohler(root="datasets/Kohler",
                             frames="middle",
                             ordering="printout_first",
                             download=True)
            # Usual interface
            sharp_frame, blurry_shot = dataset[0]
            print(sharp_frame.shape, blurry_shot.shape)
            # Convenience method to directly index the printouts and trajectories
            sharp_frame, blurry_shot = dataset.get_item(1, 1, frames="middle")
            print(sharp_frame.shape, blurry_shot.shape)
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
        frames: Union[int, str, list[Union[int, str]]] = "middle",
        ordering: str = "printout_first",
        transform: Callable = None,
        download: bool = False,
    ) -> None:
        self.root = root
        self.frames = frames
        self.ordering = ordering
        self.transform = transform

        if download:
            self.download(self.root)

    @classmethod
    def download(cls, root: Union[str, Path], remove_finished: bool = False) -> None:
        """Download the dataset.

        :param Union[str, pathlib.Path] root: Root directory of the dataset.
        :param bool remove_finished: Remove the archives after extraction.

        |sep|

        :Examples:

            Download the dataset ::

                from deepinv.datasets import Kohler
                Kohler.download("datasets/Kohler")
        """
        for url in cls.archive_urls:
            archive_name = url_basename(url)
            checksum = cls.archive_checksums[archive_name]

            # Download the archive and verify its integrity
            download_and_extract_archive(
                url,
                root,
                filename=archive_name,
                md5=checksum,
                remove_finished=remove_finished,
            )

    def __len__(self) -> int:
        return 48

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sharp frame and a blurry shot from the dataset.

        :param int index: Index of the pair.

        :return: (torch.Tensor, torch.Tensor) The sharp frame and the blurry shot.

        |sep|

        :Examples:

            Get the first sharp frame and blurry shot ::

                sharp_frame, blurry_shot = dataset[0]
        """
        if self.ordering == "printout_first":
            printout_index = index // 12 + 1
            trajectory_index = index % 12 + 1
        elif self.ordering == "trajectory_first":
            printout_index = index % 12 + 1
            trajectory_index = index // 12 + 1
        else:
            raise ValueError(f"Unsupported ordering: {self.ordering}")
        return self.get_item(printout_index, trajectory_index, frames=self.frames)

    # While users might sometimes want to thoroughly compare their own
    # deblurred images to all the sharp frames (about 200 per blurry shot),
    # they will probably most often make the way more convenient choice of
    # comparing against a single frame per blurry shot. For this reason, the
    # method get_item accepts an additional parameter for frame selection and
    # only returns the selected frame.
    def get_item(
        self,
        printout_index: int,
        trajectory_index: int,
        frames: Union[None, int, str, list[Union[int, str]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sharp frame and a blurry shot from the dataset.

        :param int printout_index: Index of the printout.
        :param int trajectory_index: Index of the trajectory.
        :param Union[None, int, str, list[Union[int, str]]] frames: Can be the frame number, "first", "middle", "last", or "all". If a list is provided, the method will return a list of sharp frames. By default, it uses the value provided in the constructor.

        :return: (torch.Tensor, Union[torch.Tensor, list[torch.Tensor]]) The sharp frame(s) and the blurry shot.

        |sep|

        :Examples:

            Get the first (middle) sharp frame and blurry shot ::

                sharp_frame, blurry_shot = dataset.get_item(1, 1, frame="middle")

            Get the list of all sharp frames and the blurry shot ::

                sharp_frames, blurry_shot = dataset.get_item(1, 1, frame="all")

            Query a list of specific frames and the blurry shot ::

                sharp_frames, blurry_shot = dataset.get_item(1, 1, frame=[1, "middle", 199])
        """
        blurry_shot = self.get_blurry_shot(printout_index, trajectory_index)

        if frames is None:
            frames = self.frames

        if frames == "all" or isinstance(frames, list):
            if frames == "all":
                frames = range(
                    1, self.get_frame_count(printout_index, trajectory_index) + 1
                )
            sharp_frames = [
                self.get_sharp_frame(printout_index, trajectory_index, frame_index)
                for frame_index in frames
            ]
            return sharp_frames, blurry_shot
        else:
            frame_index = self.select_frame(
                printout_index, trajectory_index, frame=frames
            )
            sharp_frame = self.get_sharp_frame(
                printout_index, trajectory_index, frame_index
            )
            return sharp_frame, blurry_shot

    def get_sharp_frame(
        printout_index: int, trajectory_index: int, frame_index: int
    ) -> Union[torch.Tensor, Image.Image, any]:
        path = join(
            self.root,
            f"Image{printout_index}",
            f"Kernel{trajectory_index}",
            f"GroundTruth{printout_index}_{trajectory_index}_{frame_index}.png",
        )
        sharp_frame = Image.open(path)
        if self.transform is not None:
            sharp_frame = self.transform(sharp_frame)
        return sharp_frame

    def get_blurry_shot(
        printout_index: int, trajectory_index: int
    ) -> Union[torch.Tensor, Image.Image, any]:
        path = join(self.root, f"Blurry{printout_index}_{trajectory_index}.png")
        blurry_shot = Image.open(path)
        if self.transform is not None:
            blurry_shot = self.transform(blurry_shot)
        return blurry_shot

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
                raise ValueError(f"Unsupported frame selection: {frame}")

        return frame_index

    @classmethod
    def get_frame_count(cls, printout_index: int, trajectory_index: int) -> int:
        index = (printout_index, trajectory_index)

        if index in cls.frame_count_table:
            count = cls.frame_count_table[index]
        else:
            count = cls.frame_count_table["others"]

        return count
