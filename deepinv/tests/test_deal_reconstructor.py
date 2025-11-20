import sys
import types
import torch
import torch.nn as nn

from deepinv.models import DEAL
from deepinv.physics import Denoising


class DummyInnerDEAL(nn.Module):
    """
    Tiny fake version of the external deal.DEAL class,
    just enough so that our wrapper and the test can run.
    """

    def __init__(self, color: bool = False, *args, **kwargs):
        # accept `color` and ignore it
        super().__init__()
        # one simple conv so we have some parameters in state_dict
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def solve_inverse_problem(
        self,
        y,
        H,
        Ht,
        sigma,
        lmbda,
        max_iter,
        x_init,
        verbose: bool = False,
        path: bool = False,
    ):
        # For the test we ignore H, Ht, sigma, lmbda, max_iter, etc.
        # Just return something with the correct shape.
        return x_init + y


def fake_load(path, map_location=None, weights_only=False):
    """
    Return a dummy state_dict so load_state_dict() does not fail.
    """
    return {"state_dict": DummyInnerDEAL().state_dict()}


def test_deal_model_runs(monkeypatch):
    import deepinv.models.deal as deal_mod

    # 1) Replace the external 'deal' library by our tiny dummy class
    deal_mod.deal_lib = types.SimpleNamespace(DEAL=DummyInnerDEAL)

    # 2) Replace torch.load used inside deepinv.models.deal.DEAL.__init__
    monkeypatch.setattr(deal_mod.torch, "load", fake_load)

    # 3) Create the wrapper model
    model = DEAL(
        checkpoint_path="dummy.pth",
        sigma=25.0,
        lam=10.0,
        max_iter=1,
        auto_scale=False,
    )

    # 4) Simple DeepInverse physics (denoising)
    physics = Denoising()

    # 5) Fake measurement
    y = torch.randn(1, 1, 32, 32)

    # 6) Run the forward pass
    x_hat = model(y, physics)

    # 7) Just check that the output shape is correct
    assert x_hat.shape == y.shape
