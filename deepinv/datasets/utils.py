import hashlib
import os
import shutil
import zipfile

import requests
from tqdm.auto import tqdm


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
    # Ensure the directory containing `save_path`` exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # `stream=True` to avoid loading in memory an entire file, instead get a chunk
    # useful when downloading huge file
    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    # use tqdm progress bar to follow progress on downloading zipfile
    with tqdm.wrapattr(response.raw, "read", total=file_size) as r_raw:
        with open(save_path, "wb") as file:
            # shutil.copyfileobj doesn't require the whole file in memory before writing in a file
            # https://requests.readthedocs.io/en/latest/user/quickstart/#raw-response-content
            shutil.copyfileobj(r_raw, file)
    del response


def extract_zipfile(file_path, extract_dir):
    """Extract a local zipfile."""
    # Open the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        # progress bar on the total number of files to be extracted
        # since files may be very huge or very small, the extraction time vary per file
        # thus the progres bar will not move linearly with time
        for file_to_be_extracted in tqdm(zip_ref.infolist(), desc="Extracting"):
            zip_ref.extract(file_to_be_extracted, extract_dir)
