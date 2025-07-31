"""Signal processing utilities"""


def normalize(inp, *, mode):
    r"""
    Normalize a batch of signals between zero and one.

    :param torch.Tensor inp: the input signal to normalize, it should be of shape (B, *).
    :param str mode: the normalization, either 'min_max' for min-max normalization or 'clip' for clipping. Note that min-max normalization of constant signals is ill-defined and left unspecified.
    :return: the normalized batch of signals.
    """
    inp = inp.clone()
    if mode == "min_max":
        shape = inp.shape
        inp = inp.reshape(shape[0], -1)
        mini = inp.min(1)[0]
        maxi = inp.max(1)[0]
        idx = mini < maxi
        mini = mini[idx].unsqueeze(1)
        maxi = maxi[idx].unsqueeze(1)
        inp[idx, :] = (inp[idx, :] - mini) / (maxi - mini)
        inp = inp.reshape(shape)
    elif mode == "clip":
        inp = inp.clamp(min=0.0, max=1.0)
    else:
        raise ValueError("mode has to be either 'min_max' or 'clip'.")
    return inp
