import shutil

import PIL
import pytest

from deepinv.datasets import DIV2K, Urban100HR, CBSD68


@pytest.fixture
def download_div2k():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "DIV2K"

    # Download div2K dataset
    DIV2K(tmp_data_dir, mode="val", download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_div2k_dataset(download_div2k):
    """Check that DIV2K/DIV2K_train_HR contains 800 items."""
    val_dataset = DIV2K(download_div2k, mode="val", download=False)
    assert (
        len(val_dataset) == 100
    ), f"Val dataset should have been of len 100, instead got {len(val_dataset)}."
    assert (
        type(val_dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_Urban100():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "Urban100"

    # Download div2K dataset
    Urban100HR(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_urban100_dataset(download_Urban100):
    """Check that dataset contains 800 items."""
    dataset = Urban100HR(download_Urban100, download=False)
    assert (
        len(dataset) == 100
    ), f"Dataset should have been of len 100, instead got {len(dataset)}."
    assert (
        type(dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."


@pytest.fixture
def download_CBSD68():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "CBSD68"

    # Download CBSD raw dataset from huggingface
    CBSD68(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_CBSD68_dataset(download_CBSD68):
    """Check that dataset contains 68 PIL images."""
    dataset = CBSD68(download_CBSD68, download=False)
    assert (
        len(dataset) == 68
    ), f"Dataset should have been of len 68, instead got {len(dataset)}."
    assert (
        type(dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."
