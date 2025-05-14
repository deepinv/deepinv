import torch
import pandas as pd
from deepinv.datasets import LidcIdriSliceDataset

from unittest.mock import patch


def test_lidc():
    dummy_df = pd.DataFrame(columns=["Modality", "Subject ID", "File Location"])
    with patch("os.path.isdir", lambda *args, **kwargs: True):
        with patch("os.path.exists", lambda *args, **kwargs: True):
            with patch("pandas.read_csv", lambda *args, **kwargs: dummy_df):
                dataset = LidcIdriSliceDataset(root="/dummy")
                assert len(dataset) == 0
