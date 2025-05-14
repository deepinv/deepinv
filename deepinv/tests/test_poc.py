import torch
import pandas as pd
from deepinv.datasets import LidcIdriSliceDataset
from unittest.mock import patch
import pydicom
import numpy as np
import os
import pytest

def test_lidc():
    # 1. Empty csv
    dummy_df = pd.DataFrame(columns=["Modality", "Subject ID", "File Location"])

    with patch("os.path.isdir", return_value=True), \
         patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=dummy_df):

        dataset = LidcIdriSliceDataset(root="/dummy")
        assert len(dataset) == 0
        with pytest.raises(Exception):
            _ = dataset[0]

    # 2. Non-empty csv
    data = [
        ["CT", "Dummy_ID_1", "/dummy/Scan1"],
        ["CT", "Dummy_ID_2", "/dummy/Scan2"],
        ["CT", "Dummy_ID_3", "/dummy/Scan3"],
    ]
    dummy_df = pd.DataFrame(data, columns=["Modality", "Subject ID", "File Location"])
    # Generated using pydicomgenerator
    # https://github.com/sjoerdk/dicomgenerator
    dummy_dicom = pydicom.dcmread(
        os.path.join(os.path.dirname(__file__), "dicomgenerator_dummy.dcm")
    )
    # NOTE: dicomgenerator_dummy.dcm lacks a TransferSyntaxUID attribute.
    # We monkey patch it to make the test work.
    dummy_dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # NOTE: In lidc_idri, dcmread is imported from pydicom and stored to a variable.
    # This means that it cannot be mocked by patching pydicom.dcmread. Instead,
    # we patch the variable from the lidc_module directly.
    with patch("os.path.isdir", return_value=True), \
         patch("os.path.exists", return_value=True), \
         patch("pandas.read_csv", return_value=dummy_df), \
         patch("os.listdir", return_value=["Slice1.dcm", "Slice2.dcm"]), \
         patch("deepinv.datasets.lidc_idri.dcmread", return_value=dummy_dicom):
        dataset = LidcIdriSliceDataset(root="/dummy")
        assert len(dataset) == 6
        im = dataset[0]
        assert isinstance(im, np.ndarray)
        assert im.ndim == 2

        # Test Hounsfield units
        dataset = LidcIdriSliceDataset(root="/dummy", hounsfield_units=True)
        assert len(dataset) == 6
        im = dataset[0]
        assert isinstance(im, np.ndarray)
        assert im.ndim == 2

        # Test transform
        called = False
        def transform(x):
            nonlocal called
            called = True
            return x
        dataset = LidcIdriSliceDataset(root="/dummy", transform=transform)
        im = dataset[0]
        assert called
