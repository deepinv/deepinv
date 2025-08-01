"""Signal processing utilities"""

import torch


def normalize_signal(inp, *, mode):
    r"""
    Normalize a batch of signals between zero and one.

    :param torch.Tensor inp: the input signal to normalize, it should be of shape (B, *).
    :param str mode: the normalization, either 'min_max' for min-max normalization or 'clip' for clipping. Note that min-max normalization of constant signals is ill-defined and here it amounts to mapping the constant value to the closest value between zero and one (which is equivalent to clipping).
    :return: the normalized batch of signals.
    """
    if mode == "min_max":
        # Compute the minimum and maximum intensity of the batched signals
        non_batched_dims = list(range(1, inp.ndim))
        minimum_intensity = inp.amin(dim=non_batched_dims, keepdim=False)
        maximum_intensity = inp.amax(dim=non_batched_dims, keepdim=False)

        # Clone the signal to avoid input mutations
        inp = inp.clone()

        # The indices corresponding to the non-constant batched signals
        indices = maximum_intensity != minimum_intensity

        # Prepare the tensors for broadcasting
        shape = (-1,) + (1,) * len(non_batched_dims)
        minimum_intensity = minimum_intensity.view(*shape)
        maximum_intensity = maximum_intensity.view(*shape)

        # Rescale the non-constant batched signals between zero and one
        inp[indices] -= minimum_intensity[indices]
        inp[indices] /= maximum_intensity[indices] - minimum_intensity[indices]

        # The indices corresponding to the constant batched signals
        indices = torch.logical_not(indices)

        # Clamp constant batched signals between zero and one
        inp[indices] = inp[indices].clamp(min=0.0, max=1.0)
    elif mode == "clip":
        # Clamp every batched signal between zero and one
        inp = inp.clamp(min=0.0, max=1.0)
    else:  # pragma: no cover
        raise ValueError(
            f"Unsupported normalization mode: {mode}. Supported modes are 'min_max' and 'clip'."
        )

    return inp
