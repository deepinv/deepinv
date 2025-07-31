"""Signal processing utilities"""


def rescale_img(im, rescale_mode="min_max"):
    r"""
    Rescale an image tensor.

    :param torch.Tensor im: the image to rescale.
    :param str rescale_mode: the rescale mode, either 'min_max' or 'clip'.
    :return: the rescaled image.
    """
    img = im.clone()
    if rescale_mode == "min_max":
        shape = img.shape
        img = img.reshape(shape[0], -1)
        mini = img.min(1)[0]
        maxi = img.max(1)[0]
        idx = mini < maxi
        mini = mini[idx].unsqueeze(1)
        maxi = maxi[idx].unsqueeze(1)
        img[idx, :] = (img[idx, :] - mini) / (maxi - mini)
        img = img.reshape(shape)
    elif rescale_mode == "clip":
        img = img.clamp(min=0.0, max=1.0)
    else:
        raise ValueError("rescale_mode has to be either 'min_max' or 'clip'.")
    return img
