import shutil

import PIL
import pytest

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
    NBUDataset,
)


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
