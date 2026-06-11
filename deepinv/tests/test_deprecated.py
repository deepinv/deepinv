import pytest
import deepinv as dinv
import torch


def test_deprecated_functions():
    with pytest.warns(DeprecationWarning):
        dinv.utils.rescale_img(torch.randn(3, 16, 32), rescale_mode="min_max")

    with pytest.warns(DeprecationWarning):
        dinv.utils.plot_inset([torch.ones(2, 1, 10, 10)], show=False)
