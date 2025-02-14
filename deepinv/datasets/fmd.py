from typing import Any, Callable, List, NamedTuple
import os
import re

from PIL import Image
import requests
import torch

from deepinv.datasets.utils import (
    download_archive,
    extract_tarball,
)


class FMD(torch.utils.data.Dataset):
    """Dataset for `Fluorescence Microscopy Denoising <https://github.com/yinhaoz/denoising-fluorescence>`_.

    | 1) The Fluorescence Microscopy Denoising (FMD) dataset is dedicated to
    | Poisson-Gaussian denoising.
    | 2) The dataset consists of 12,000 real fluorescence microscopy images
    | obtained with commercial confocal, two-photon, and wide-field microscopes
    | and representative biological samples such as cells, zebrafish,
    | and mouse brain tissues.
    | 3) Image averaging is used to effectively obtain ground truth images
    | and 60,000 noisy images with different noise levels.


    **Raw data file structure:** ::

        self.root --- Confocal_BPAE_B  --- avg16 --- 1  --- HV110_P0500510000.png
                   |                    |         |      |
                   |                    |         |      -- HV110_P0500510049.png
                   |                    |         -- 20
                   |                    -- avg2
                   |                    -- avg4
                   |                    -- avg8
                   |                    -- gt
                   |                    -- raw
                   -- ...
                   -- WideField_BPAE_R --- ...
                   -- Confocal_BPAE_G.tar
                   |
                   -- WideField_BPAE_R.tar

    | 1) There are 12 image types :
    | Confocal_BPAE_B, Confocal_BPAE_G, Confocal_BPAE_R, Confocal_FISH, Confocal_MICE
    | TwoPhoton_BPAE_B, TwoPhoton_BPAE_G, TwoPhoton_BPAE_R, TwoPhoton_MICE
    | WideField_BPAE_B, WideField_BPAE_G, WideField_BPAE_R
    | 2) Each image type has its own folder.
    | 3) Each folder contains 6 subfolders- : gt, raw, avg2, avg4, avg8 and avg16.
    | 4) gt contains clean images, the others have different noise levels applied to images.
    | 5) Each subfolder has 20 subsubfolders, corresponding to the "field of view".
    | 6) Each subsubfolder has the same 50 png file names of size (512, 512).
    | 7) 12 type of img x 5 levels of noise x 20 "fov" x 50 img = 60 000 noisy img

    :param str root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param List[str] img_types: Types of microscopy image among 12.
    :param List[int] noise_levels: Level of noises applied to the image among [1, 2, 4, 8, 16].
    :param List[int] fovs: "Field of view", value between 1 and 20.
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    :param Callable transform:: (optional) A function/transform that takes in a noisy PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
    :param Callable target_transform: (optional) A function/transform that takes in a clean PIL image
        and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instantiate dataset and download raw data from the Internet

    ::
        import shutil
        from deepinv.datasets import FMD
        img_types = ["TwoPhoton_BPAE_R"]
        dataset = FMD(root="fmd", img_types=img_types, download=True)  # download raw data at root and load dataset
        print(len(dataset))                                            # check that we have 5000 images
        shutil.rmtree("fmd")                                           # remove raw data from disk

    """

    gdrive_ids = {
        "Confocal_BPAE_B.tar": "1juaumcGn5QlFRXRQyrqfbZBhF7oX__iW",
        "Confocal_BPAE_G.tar": "1Zofz11VmI1JfRIMF7rq40RVjpzM6A9vg",
        "Confocal_BPAE_R.tar": "1QoD_vMvFdFg7yREfen3t-SGLFcnLg9YQ",
        "Confocal_FISH.tar": "1SxmsythWfxnfKJfGWpT_7Adebi8jUK98",
        "Confocal_MICE.tar": "11aflcrcatFRkv7EabjWjdlpT0DYRbUDZ",
        "TwoPhoton_BPAE_B.tar": "1yVD_H_ZfNNSma5vtHZM_DTnSv1Bo1tfk",
        "TwoPhoton_BPAE_G.tar": "125nqTfQQG1-YVUs256b2vTwt4aUNCgBt",
        "TwoPhoton_BPAE_R.tar": "1rwxG6LYcKeiBKNT3Oq9lvwKu8mV3rz9P",
        "TwoPhoton_MICE.tar": "1lhsFAlXsXk26yqHzT0_-3R8MUb7G0NVa",
        "WideField_BPAE_B.tar": "19rl8zFzfXIZ2drgodCGutLPLzL4kJq6d",
        "WideField_BPAE_G.tar": "1H67O6GqIkIlQSX-n0vfMWGPwmd4zOHQr",
        "WideField_BPAE_R.tar": "19HXb2Ftrb-M7Lr9ZlHWMcnNT0Sbu85YL",
    }

    class NoisySampleIdentifier(NamedTuple):
        """Data structure for identifying noisy data sample files.

        :param str img_type: Foldername corresponding to one type of image among 12.
        :param str noise_dirname: Foldername corresponding to one level of noise,
            'raw' - level 1, 'avg2' - 2, 'avg4' - 4, 'avg8' - 8, 'avg16' - 16
        :param int fov: Field of view, value between 1 and 20.
        :param str fname: Filename of a png file containing 1 noisy image.
        """

        img_type: str
        noise_dirname: str
        fov: int
        fname: str

    def __init__(
        self,
        root: str,
        img_types: List[str],
        noise_levels: List[int] = [1, 2, 4, 8, 16],
        fovs: List[int] = list(range(1, 20 + 1)),
        download: bool = False,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        self.root = root
        self.img_types = img_types
        self.noise_levels = noise_levels
        self.fovs = fovs
        self.transform = transform
        self.target_transform = target_transform

        all_types = [
            "TwoPhoton_BPAE_R",
            "TwoPhoton_BPAE_G",
            "TwoPhoton_BPAE_B",
            "TwoPhoton_MICE",
            "Confocal_MICE",
            "Confocal_BPAE_R",
            "Confocal_BPAE_G",
            "Confocal_BPAE_B",
            "Confocal_FISH",
            "WideField_BPAE_R",
            "WideField_BPAE_G",
            "WideField_BPAE_B",
        ]
        if img_types is None or not all(
            img_type in all_types for img_type in img_types
        ):
            raise ValueError(
                f"Wrong image type. Set `img_types` argument with these values: {all_types}"
            )

        all_noise_levels = [1, 2, 4, 8, 16]
        if not all(level in all_noise_levels for level in noise_levels):
            raise ValueError(f"Wrong noise level. Available levels: {all_noise_levels}")

        ### DOWNLOAD -------------------------------------------------------------------

        if download:
            if not os.path.isdir(self.root):
                os.makedirs(self.root)

            for img_type in self.img_types:
                filename = img_type + ".tar"
                gdrive_id = self.gdrive_ids[filename]

                ## We need to access the content of a html file to retrieve information
                ## Which will be needed to download the archive ------------------------

                # URL to fetch the initial HTML content
                url_initial = (
                    f"https://docs.google.com/uc?export=download&id={gdrive_id}"
                )
                response_initial = requests.get(url_initial)
                html_content = response_initial.text

                # Extract UUID using regular expression
                uuid_match = re.search(r'name="uuid" value="([^"]+)"', html_content)

                if uuid_match:
                    uuid = uuid_match.group(1)
                else:
                    raise ValueError(
                        "UUID not found in the HTML content, can't download dataset"
                    )

                # URL to download the file
                url_download = "https://drive.usercontent.google.com/"
                url_download += f"download?id={gdrive_id}&export=download&authuser=0&confirm=t&uuid={uuid}"

                # download tar file from the Internet and save it locally
                download_archive(
                    url=url_download, save_path=os.path.join(self.root, filename)
                )

                # extract local tar file
                extract_tarball(os.path.join(self.root, filename), self.root)

        ### GET DATA SAMPLE IDENTIFIERS -----------------------------------------------

        # should contain all the information to load a data sample from the storage
        self.noisy_sample_identifiers = []

        for img_type in img_types:
            for noise_level in noise_levels:
                if noise_level == 1:
                    noise_dirname = "raw"
                else:  # noise_level is in [2, 4, 8, 16]
                    noise_dirname = f"avg{noise_level}"
                for fov in fovs:
                    folder_path = os.path.join(
                        self.root, img_type, noise_dirname, str(fov)
                    )
                    try:
                        noisy_img_list = sorted(os.listdir(folder_path))
                    except FileNotFoundError:
                        raise RuntimeError(
                            f"Data folder {folder_path} doesn't exist, please set `download=True`"
                        )
                    for fname in noisy_img_list:
                        if fname.endswith(".png"):
                            self.noisy_sample_identifiers.append(
                                self.NoisySampleIdentifier(
                                    img_type, noise_dirname, fov, fname
                                )
                            )

    def __len__(self) -> int:
        return len(self.noisy_sample_identifiers)

    def __getitem__(self, idx: int) -> Any:
        img_type, noise_dirname, fov, fname = self.noisy_sample_identifiers[idx]
        noisy_img_path = os.path.join(
            self.root, img_type, noise_dirname, str(fov), fname
        )
        clean_img_path = os.path.join(self.root, img_type, "gt", str(fov), "avg50.png")
        # PIL Image
        noisy_img = Image.open(noisy_img_path)
        clean_img = Image.open(clean_img_path)

        if self.transform is not None:
            noisy_img = self.transform(noisy_img)
        if self.target_transform is not None:
            clean_img = self.target_transform(clean_img)
        return noisy_img, clean_img
