import shutil

import PIL
import pytest

from deepinv.datasets import DIV2K, Urban100HR


@pytest.fixture
def download_div2k():
    """Downloads dataset for tests and removes it after test executions."""
    tmp_data_dir = "DIV2K"

    # Download div2K dataset
    DIV2K(tmp_data_dir, download=True)

    # This will return control to the test function
    yield tmp_data_dir

    # After the test function complete, any code after the yield statement will run
    shutil.rmtree(tmp_data_dir)


def test_load_div2k_dataset(download_div2k):
    """Check that DIV2K/DIV2K_train_HR contains 800 items."""
    dataset = DIV2K(download_div2k, download=False)
    assert len(dataset) == 800


def test_load_div2K_img(download_div2k):
    """Check that DIV2K/DIV2K_train_HR contains images."""
    dataset = DIV2K(download_div2k, download=False)
    assert type(dataset[0]) == PIL.PngImagePlugin.PngImageFile


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


def test_load_urban100_img(download_Urban100):
    """Check that dataset contains images."""
    dataset = Urban100HR(download_Urban100, download=False)
    assert (
        type(dataset[0]) == PIL.PngImagePlugin.PngImageFile
    ), "Dataset image should have been a PIL image."
