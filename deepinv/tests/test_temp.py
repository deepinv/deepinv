import pytest
import torch
import deepinv as dinv

@pytest.fixture
def image(imsize):
    x = torch.zeros(imsize).unsqueeze(0)
    x[..., x.shape[-2] // 2 - 2:x.shape[-2] // 2 + 2, x.shape[-1] // 2 - 2:x.shape[-1] // 2 + 2]
    return x

def test_update_parameters(image):
    dinv.utils.plot