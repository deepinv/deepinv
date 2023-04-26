import git
import requests
import shutil
import os
import zipfile


def get_git_root():
    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def download_dataset(dataset_name, data_dir, url=None):
    dataset_dir = data_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if url is None:
        url = f"https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fdatasets&files={dataset_name}.zip"
        try:
            with open(str(dataset_dir / dataset_name) + ".zip", "wb") as f:
                request = requests.get(url)
                f.write(request.content)
            with zipfile.ZipFile(str(dataset_dir / dataset_name) + ".zip") as zip_ref:
                zip_ref.extractall(str(data_dir))
            print(f"{dataset_name} dataset downloaded in {data_dir}")
        except:
            print(f"{dataset_name} dataset downloading failed")


def download_degradation(name, data_dir, url=None):
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://mycore.core-cloud.net/index.php/s/9EzDqcJxQUJKYul/download?path=%2Fdatasets&files={name}"
    try:
        with requests.get(url, stream=True) as r:
            with open(str(data_dir / name), "wb") as f:
                shutil.copyfileobj(r.raw, f)
        print(f"{name} degradation downloaded in {data_dir}")
    except:
        print(f"{name} degradation downloading failed")
