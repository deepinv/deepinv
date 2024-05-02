import hashlib
import os
import shutil
import zipfile

import requests


def check_path_is_a_folder(folder_path: str) -> bool:
    # Check if `folder_path` is pointing to a directory
    if not os.path.isdir(folder_path):
        return False
    # Check if everything inside of the directory is a file
    return all(
        os.path.isfile(os.path.join(folder_path, filename))
        for filename in os.listdir(folder_path)
    )


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    """From https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py#L35"""
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def calculate_md5_for_folder(folder_path: str):
    """Compute the hash of all files in a folder then compute the hash of the folder.

    Folder will be considered as empty if it is not strictly containing files.
    """
    md5_folder = hashlib.md5()
    if check_path_is_a_folder(folder_path):
        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            md5_folder.update(calculate_md5(file_path).encode())
    return md5_folder.hexdigest()


def download_zipfile(url, save_path):
    """Download zipfile from the Internet."""
    # `stream=True` to avoid loading in memory an entire file, instead get a chunk
    # useful when downloading huge file
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        # shutil.copyfileobj doesn't require the whole file in memory before writing in a file
        # https://requests.readthedocs.io/en/latest/user/quickstart/#raw-response-content
        shutil.copyfileobj(response.raw, file)
    del response


def extract_zipfile(file_path, extract_dir):
    """Extract a local zipfile."""
    # Open the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        # Extract all the contents of the zip file to the specified dir
        zip_ref.extractall(extract_dir)
