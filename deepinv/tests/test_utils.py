import deepinv
import torch
import pytest

import matplotlib.pyplot as plt


@pytest.fixture
def tensorlist():
    x = torch.ones((1, 1, 2, 2))
    y = torch.ones((1, 1, 2, 2))
    x = deepinv.utils.TensorList([x, x])
    y = deepinv.utils.TensorList([y, y])
    return x, y


def test_tensordict_sum(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2)) * 2
    z1 = deepinv.utils.TensorList([z, z])
    z = x + y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()
    z = y + x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_mul(tensorlist):
    x, y = tensorlist
    alpha = 1.0
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x * alpha
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()
    z = alpha * x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_scalar_mul(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x * y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_div(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x / y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_sub(tensorlist):
    x, y = tensorlist
    z = torch.zeros((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x - y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_neg(tensorlist):
    x, y = tensorlist
    z = -torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = -x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_append(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z, z, z])
    z = x.append(y)
    assert (z1[0] == z[0]).all() and (z1[-1] == z[-1]).all()


def test_plot():
    for c in range(1, 5):
        x = torch.ones((1, c, 2, 2))
        imgs = [x, x]
        deepinv.utils.plot(imgs, titles=["a", "b"], show=False)
        deepinv.utils.plot(x, titles="a", show=False)
        deepinv.utils.plot(imgs, show=False)


def test_plot_inset():
    # Plots a batch of images with a checkboard pattern, with different inset locations
    x = torch.ones(2, 1, 100, 100)

    for i in range(0, 100, 10):
        x[:, :, :, i : i + 5] = 0
        x[:, :, i : i + 5, :] = 0

    deepinv.utils.plot_inset(
        [x],
        titles=["a"],
        labels=["a"],
        inset_loc=((0, 0.5), (0.5, 0.5)),
        show=False,
        save_fn="temp.png",
    )
