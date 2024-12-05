import shutil, os

import PIL
import pytest
import torch

from deepinv.datasets import (
    DIV2K,
    Urban100HR,
    Set14HR,
    CBSD68,
    LsdirHR,
    FMD,
    FastMRISliceDataset, SimpleFastMRISliceDataset
)
from deepinv.datasets.utils import download_archive
from deepinv.utils.demo import get_image_dataset_url


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
def download_cbsd68():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "CBSD68"

    # Download CBSD raw dataset from huggingface
    CBSD68(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
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

def test_SimpleFastMRISliceDataset(download_simplefastmri):
    dataset = SimpleFastMRISliceDataset(
        root_dir=download_simplefastmri,
        anatomy="knee",
        train=True,
        train_percent=1.,
        download=False
    )
    x = dataset[0]
    x2 = dataset[1]
    assert x.shape == (2, 320, 320)
    assert not torch.all(x == x2)
    assert len(dataset) == 973

@pytest.fixture
def download_fastmri():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "fastmri"
    file_name = "brain_multicoil_train_0.h5"

    # Download single FastMRI volume
    os.makedirs(tmp_data_dir, exist_ok=True)
    url = get_image_dataset_url(file_name, None)
    download_archive(url, f"{tmp_data_dir}/{file_name}")

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)

def test_FastMRISliceDataset(download_fastmri):
    dataset = FastMRISliceDataset(
        root=download_fastmri,
        slice_index="all",
    )
    target1, kspace1 = dataset[0]
    target2, kspace2 = dataset[1]

    assert target1.shape == (320, 320)
    assert kspace1.shape == (2, 20, 512, 256)
    assert not torch.all(target1 == target2)
    assert not torch.all(kspace1 == kspace2)

    subset = dataset.save_simple_dataset(f"{download_fastmri}/temp_simple.pt")
    x = subset[0]

    assert x.shape == (2, 320, 320)