import gzip

import numpy as np
import pytest
import torch

import deepinv.datasets.brainweb as brainweb_module
from deepinv.datasets import BrainWebDataset, BrainWebLesion


@pytest.fixture
def brainweb_root(tmp_path, monkeypatch):
    monkeypatch.setattr(BrainWebDataset, "image_size", (15, 17, 19))
    monkeypatch.setattr(BrainWebDataset, "voxel_size", (1.0, 1.0, 1.0))

    labels = np.full(BrainWebDataset.image_size, 2, dtype=np.uint8)
    labels[0, 0, 0] = 0
    labels[0, 0, 1] = 1
    labels[0, 0, 2] = 3
    labels[0, 0, 3] = 7
    labels[0, 0, 4] = 10
    labels[0, 0, 5] = 11

    path = tmp_path / "subject_04.raw_byte.bin.gz"
    with gzip.open(path, "wb") as file:
        file.write(labels.tobytes())
    return tmp_path, labels


def test_brainweb_activity_and_attenuation(brainweb_root):
    root, labels = brainweb_root
    dataset = BrainWebDataset(
        root=root,
        subject_ids=4,
        activity_levels={"grey matter": 5, "white_matter": 2},
    )

    activity, params = dataset[0]
    attenuation = params["attenuation"]

    assert len(dataset) == 1
    assert activity.shape == (1,) + labels.shape
    assert activity.dtype == torch.float32
    assert activity[0, 1, 1, 1] == 5
    assert activity[0, 0, 0, 2] == 2
    assert activity[0, 0, 0, 0] == 0
    assert attenuation.shape == activity.shape
    assert attenuation[0, 0, 0, 1] == pytest.approx(0.0975)
    assert attenuation[0, 0, 0, 3] == pytest.approx(0.13)
    assert attenuation[0, 0, 0, 4] == pytest.approx(0.13)
    assert attenuation[0, 0, 0, 5] == pytest.approx(0.13)
    dataset.check_dataset()


def test_brainweb_explicit_lesion_and_mask(brainweb_root):
    root, _ = brainweb_root
    center = (7, 8, 9)
    dataset = BrainWebDataset(
        root=root,
        lesions=[
            BrainWebLesion(
                diameter_mm=2,
                activity=0,
                center_voxel=center,
            )
        ],
    )

    activity, params = dataset[0]
    lesion_mask = params["lesion_mask"]

    assert activity[(0,) + center] == 0
    assert lesion_mask.dtype == torch.uint8
    assert lesion_mask[(0,) + center] == 1
    assert torch.count_nonzero(lesion_mask == 1) == 7
    torch.testing.assert_close(
        params["lesion_centers"], torch.tensor([center], dtype=torch.float32)
    )


def test_brainweb_random_lesions_are_reproducible(brainweb_root):
    root, _ = brainweb_root
    lesions = [
        BrainWebLesion(diameter_mm=2, activity=10),
        BrainWebLesion(diameter_mm=2, activity=0),
    ]
    dataset = BrainWebDataset(root=root, lesions=lesions, seed=123)

    activity_1, params_1 = dataset[0]
    activity_2, params_2 = dataset[0]

    torch.testing.assert_close(activity_1, activity_2)
    torch.testing.assert_close(params_1["lesion_mask"], params_2["lesion_mask"])
    torch.testing.assert_close(params_1["lesion_centers"], params_2["lesion_centers"])
    assert torch.count_nonzero(params_1["lesion_mask"] == 1) > 0
    assert torch.count_nonzero(params_1["lesion_mask"] == 2) > 0
    assert set(torch.unique(params_1["lesion_mask"]).tolist()) == {0, 1, 2}


def test_brainweb_downloads_only_requested_subjects(tmp_path, monkeypatch):
    monkeypatch.setattr(BrainWebDataset, "image_size", (3, 4, 5))
    monkeypatch.setattr(BrainWebDataset, "voxel_size", (1.0, 1.0, 1.0))
    requested_urls = []

    def fake_download(url, save_path):
        requested_urls.append(url)
        labels = np.zeros(BrainWebDataset.image_size, dtype=np.uint8)
        with gzip.open(save_path, "wb") as file:
            file.write(labels.tobytes())

    monkeypatch.setattr(brainweb_module, "download_archive", fake_download)
    dataset = BrainWebDataset(
        root=tmp_path, subject_ids=[4, 6], download=True, activity_levels={}
    )

    assert len(dataset) == 2
    assert len(requested_urls) == 2
    assert "subject04_crisp" in requested_urls[0]
    assert "subject06_crisp" in requested_urls[1]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "subject_04.raw_byte.bin.gz",
        "subject_06.raw_byte.bin.gz",
    ]


def test_brainweb_missing_subject_fails(tmp_path):
    with pytest.raises(RuntimeError, match="download=True"):
        BrainWebDataset(root=tmp_path, subject_ids=4)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"subject_ids": 7}, ValueError),
        ({"subject_ids": [4, 4]}, ValueError),
        ({"activity_levels": "unknown"}, ValueError),
        ({"activity_levels": {"not_a_tissue": 1}}, ValueError),
        ({"random_lesion_tissues": []}, ValueError),
    ],
)
def test_brainweb_invalid_config(brainweb_root, kwargs, error):
    root, _ = brainweb_root
    with pytest.raises(error):
        BrainWebDataset(root=root, **kwargs)


def test_brainweb_lesion_validation():
    with pytest.raises(ValueError, match="strictly positive"):
        BrainWebLesion(diameter_mm=0, activity=1)
    with pytest.raises(ValueError, match="non-negative"):
        BrainWebLesion(diameter_mm=1, activity=-1)
