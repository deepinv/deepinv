import shutil, os

import PIL
import pytest
import torch

import torch
from torch import Tensor

from deepinv.datasets import (
    DIV2K,
    Urban100HR,
    Set14HR,
    CBSD68,
    LsdirHR,
    FMD,
    Kohler,
    FastMRISliceDataset,
    SimpleFastMRISliceDataset,
    NBUDataset,
)
from deepinv.datasets.utils import download_archive
from deepinv.utils.demo import get_image_url
from deepinv.physics.mri import MultiCoilMRI, MRI


@pytest.fixture
def download_div2k():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "DIV2K"

    # Download div2K raw dataset
    DIV2K(tmp_data_dir, mode="val", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_div2k_dataset(download_div2k):
    """Check that DIV2K/DIV2K_train_HR contains 800 PIL images."""
    val_dataset = DIV2K(download_div2k, mode="val", download=False)
    assert (
        len(val_dataset) == 100
    ), f"Val dataset should have been of len 100, instead got {len(val_dataset)}."
    assert (
        type(val_dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_urban100():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "Urban100"

    # Download Urban100 raw dataset
    Urban100HR(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_urban100_dataset(download_urban100):
    """Check that dataset contains 100 PIL images."""
    dataset = Urban100HR(download_urban100, download=False)
    assert (
        len(dataset) == 100
    ), f"Dataset should have been of len 100, instead got {len(dataset)}."
    assert (
        type(dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_set14():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "Set14"

    # Download Set14 raw dataset
    Set14HR(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.skip(reason="Set14 dataset download is temporarily unavailable.")
def test_load_set14_dataset(download_set14):
    """Check that dataset contains 14 PIL images."""
    dataset = Set14HR(download_set14, download=False)
    assert (
        len(dataset) == 14
    ), f"Dataset should have been of len 14, instead got {len(dataset)}."
    assert (
        type(dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_cbsd68(download=True):
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "CBSD68"

    # Download CBSD raw dataset from huggingface
    try:
        CBSD68(tmp_data_dir, download=download)
    except ImportError:
        download = False

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    if download:
        shutil.rmtree(tmp_data_dir)


def test_load_cbsd68_dataset(download_cbsd68):
    """Check that dataset contains 68 PIL images."""

    pytest.importorskip(
        "datasets",
        reason="This test requires datasets. It should be "
        "installed with `pip install datasets`",
    )

    dataset = CBSD68(download_cbsd68, download=False)
    assert (
        len(dataset) == 68
    ), f"Dataset should have been of len 68, instead got {len(dataset)}."
    assert (
        type(dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_Kohler():
    """Download the Köhler dataset before a test and remove it after completion."""
    root = "Kohler"
    Kohler.download(root)

    # Return the control flow to the test function
    yield root

    # Clean up the created directory
    shutil.rmtree(root)


@pytest.mark.skip(reason="Downloading Kohler dataset is unreliable for testing.")
def test_load_Kohler_dataset(download_Kohler):
    """Check that the Köhler dataset contains 48 PIL images."""
    root = download_Kohler

    dataset = Kohler(
        root=root, frames="middle", ordering="printout_first", download=False
    )
    x1, y1 = dataset.get_item(1, 1, "middle")
    x2, y2 = dataset[0]

    assert (
        len(dataset) == 48
    ), f"The dataset should have been of len 48, instead got {len(dataset)}."

    assert (
        type(x1) == PIL.PngImagePlugin.PngImageFile
    ), "The sharp frame is unexpectedly not a PIL image."

    assert (
        type(y1) == PIL.PngImagePlugin.PngImageFile
    ), "The blurry frame is unexpectedly not a PIL image."

    assert (
        type(x2) == PIL.PngImagePlugin.PngImageFile
    ), "The sharp frame is unexpectedly not a PIL image."

    assert (
        type(y2) == PIL.PngImagePlugin.PngImageFile
    ), "The blurry frame is unexpectedly not a PIL image."


def download_lsdir():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "LSDIR"

    # Download LSDIR raw dataset
    LsdirHR(tmp_data_dir, mode="val", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.skip(reason="Skipping this test for now, url links are not working")
def test_load_lsdir_dataset(download_lsdir):
    """Check that dataset contains 250 PIL images."""
    dataset = LsdirHR(download_lsdir, mode="val", download=False)
    assert (
        len(dataset) == 250
    ), f"Dataset should have been of len 250, instead got {len(dataset)}."
    assert (
        type(dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_fmd():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "FMD"

    # indicates which subsets we want to download
    types = ["TwoPhoton_BPAE_R"]

    # Download FMD raw dataset
    FMD(tmp_data_dir, img_types=types, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.mark.skip(reason="Downloading FMD dataset is unreliable for testing.")
def test_load_fmd_dataset(download_fmd):
    """Check that dataset contains 5000 noisy PIL images with its ground truths."""
    types = ["TwoPhoton_BPAE_R"]
    dataset = FMD(download_fmd, img_types=types, download=True)
    assert (
        len(dataset) == 5000
    ), f"Dataset should have been of len 5000, instead got {len(dataset)}."
    assert (
        type(dataset[0][0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_simplefastmri():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "fastmri"

    # Download simple FastMRI slice dataset
    SimpleFastMRISliceDataset(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


@pytest.fixture
def download_nbu():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "NBU"

    # Download Urban100 raw dataset
    NBUDataset(tmp_data_dir, satellite="gaofen-1", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_nbu_dataset(download_nbu):
    """Check that dataset correct length and type."""
    dataset = NBUDataset(download_nbu, satellite="gaofen-1", download=False)
    assert (
        len(dataset) == 5
    ), f"Dataset should have been of len 5, instead got {len(dataset)}."
    assert (
        isinstance(dataset[0], Tensor)
        and torch.all(dataset[0] <= 1)
        and torch.all(dataset[0] >= 0)
    ), "Dataset image should be Tensor between 0-1."


def test_SimpleFastMRISliceDataset(download_simplefastmri):
    dataset = SimpleFastMRISliceDataset(
        root_dir=download_simplefastmri,
        anatomy="knee",
        train=True,
        train_percent=1.0,
        download=False,
    )
    x = dataset[0]
    x2 = dataset[1]
    assert x.shape == (2, 320, 320)
    assert not torch.all(x == x2)
    assert len(dataset) == 2


@pytest.fixture
def download_fastmri():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "fastmri"
    file_name = "demo_fastmri_brain_multicoil.h5"

    # Download single FastMRI volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    url = get_image_url(file_name)
    download_archive(url, f"{tmp_data_dir}/{file_name}")

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_FastMRISliceDataset(download_fastmri):
    # Raw data shape
    kspace_shape = (512, 213)
    n_coils = 4
    img_shape = (213, 213)

    # Clean data shape
    rss_shape = (320, 320)

    dataset = FastMRISliceDataset(
        root=download_fastmri,
        slice_index="all",
    )
    target1, kspace1 = dataset[0]
    target2, kspace2 = dataset[1]

    assert target1.shape == (1, *img_shape)
    assert kspace1.shape == (2, n_coils, *kspace_shape)
    assert not torch.all(target1 == target2)
    assert not torch.all(kspace1 == kspace2)

    physics = MultiCoilMRI(
        mask=torch.ones(kspace_shape),
        coil_maps=torch.ones(kspace_shape, dtype=torch.complex64),
        img_size=img_shape,
    )
    rss1 = physics.A_adjoint(kspace1.unsqueeze(0), rss=True, crop=True)
    assert torch.allclose(target1.unsqueeze(0), rss1)

    if False:
        # This code can be used to test singlecoil data (e.g. singlecoil knee data)
        # But we omit this for downloading speed.
        physics = MRI(mask=torch.ones(kspace_shape), img_size=img_shape)
        mag1 = physics.A_adjoint(kspace1.unsqueeze(0), mag=True, crop=True)
        assert torch.allclose(target1.unsqueeze(0), mag1)

    subset = dataset.save_simple_dataset(f"{download_fastmri}/temp_simple.pt")
    x = subset[0]
    assert len(subset) == 16  # 16 slices
    assert x.shape == (2, *rss_shape)
