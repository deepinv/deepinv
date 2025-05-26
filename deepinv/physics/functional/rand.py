from warnings import warn
import numpy as np
import torch


def random_choice(
    a: torch.Tensor,
    size: tuple[int] = None,
    replace: bool = True,
    p: torch.Tensor = None,
    rng: torch.Generator = None,
):
    r"""
    PyTorch equivalent of `numpy.random.choice`
    :param torch.Tensor a: the 1-D input tensor
    :param size: output shape.
    :param bool replace: whether to sample with replacement.
        Default is True, meaning that a value of `a` can be selected multiple times.
    :param torch.Tensor p: the probabilities for each entry in `a`.
        If not given, the sample assumes a uniform distribution over all entries in `a`.

    :return: the generated random samples in the same device as `a`.

    |sep|

    :Examples:

    >>> import torch
    >>> from deepinv.physics.functional import random_choice
    >>> a = torch.tensor([1.,2.,3.,4.,5.])
    >>> p = torch.tensor([0,0,1.,0,0])
    >>> print(random_choice(a, 2, replace=True, p=p))
    tensor([3., 3.])
    """
    if isinstance(a, int):
        if rng is not None:
            device = rng.device
        elif p is not None:
            device = p.device
        else:
            device = torch.device("cpu")
        a = torch.arange(a, device=device)
    if a.ndim > 1:
        warn(
            f"The input must be a one-dimensional tensor, but got input of shape {a.shape}. The input will be flattened."
        )
        a = a.flatten()
    if p is None:
        if not replace:
            if np.prod(size) > a.numel():
                raise ValueError(
                    "Cannot take a larger sample than population when 'replace=False'"
                )
            else:
                indices = torch.randperm(a.size(0), generator=rng, device=a.device)[
                    : np.prod(size)
                ].view(size)
        else:
            indices = torch.randint(
                low=0, high=a.size(0), size=size, device=a.device, generator=rng
            )
    else:
        if p.ndim > 1:
            warn(
                f"The probability must be a one-dimensional tensor, but got input of shape {p.shape}. The input will be flattened."
            )
            p = p.flatten()
        if not torch.allclose(p.sum(), torch.tensor(1.0, device=p.device)):
            raise ValueError("The probabilities must sum up to 1.")
        if p.shape != a.shape:
            raise ValueError(
                "The probabilities and the input tensor should have the same shape."
            )

        indices = torch.multinomial(
            p, num_samples=np.prod(size), replacement=replace, generator=rng
        ).view(size)

    return a[indices]
