"""Signal processing utilities"""


def normalize(inp, *, mode):
    r"""
    Normalize a batch of signals between zero and one.

    :param torch.Tensor inp: the input signal to normalize, it should be of shape (B, *).
    :param str mode: the normalization, either 'min_max' for min-max normalization or 'clip' for clipping. Note that min-max normalization of constant signals is ill-defined and here 'min_max' amounts to clipping for constant signals.
    :return: the normalized batch of signals.
    """
    min_val = 0.0
    max_val = 1.0

    # NOTE: Rescaling a constant signal between zero and one is ill-defined.
    # Indeed, no matter the new scale, the signal won't have a minimum value
    # equal to zero and a maximum value equal to one. In this case, we
    # choose to rescale the signal so that its new value is the number in
    # [0, 1] that is closest to its original value. This amounts to
    # clamping it between zero and one.
    if mode == "min_max":
        # Compute batch-wise minimum and maximum values
        non_batched_dims = list(range(1, inp.ndim))
        inp_min = inp.amin(dim=non_batched_dims, keepdim=False)
        inp_max = inp.amax(dim=non_batched_dims, keepdim=False)

        # Clone the signal to avoid input mutations
        inp = inp.clone()

        # Compute indices of non-constant the batch of signals
        indices = inp_max != inp_min
        # Make inp_min and inp_max broadcastable with inp
        shape = (-1,) + (1,) * (inp.ndim - 1)
        inp_min = inp_min.view(*shape)
        inp_max = inp_max.view(*shape)
        # Rescale non-constant batched signals between zero and one
        inp[indices] -= inp_min[indices]
        inp[indices] /= inp_max[indices] - inp_min[indices]

        # Compute indices of constant batched signals
        indices = torch.logical_not(indices)
        # Clamp constant batched signals between zero and one
        inp[indices] = inp[indices].clamp(min=min_val, max=max_val)
    elif mode == "clip":
        inp = inp.clamp(min=min_val, max=max_val)
    else:
        raise ValueError("rescale_mode has to be either 'min_max' or 'clip'.")

    return inp
