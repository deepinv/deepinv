from __future__ import annotations
from torch import Tensor


def plot_napari(*x: Tensor, screenshot: bool = False):  # pragma: no cover
    """View 2D images or 3D volumes in napari.

    Opens an interactive `napari <https://napari.org>`_ viewer displaying the
    provided tensors side by side in a grid. This is useful for inspecting
    volumetric data such as 3D microscopy stacks, which cannot be easily
    rendered with :func:`deepinv.utils.plot`.

    Requires `napari` to be installed. Install it with ``pip install "napari[all]"``.

    .. note::
        This function opens an interactive window and therefore requires a
        display. Pass ``screenshot=True`` to render off-screen and return a
        `PIL.Image.Image` instead, which is useful for documentation
        or when running headless.

    :param torch.Tensor x: tensors passed as args, accepts 1 to 6.
        Each must be either a 2D image of shape `(1, 1, H, W)` or a 3D volume of
        shape `(1, 1, D, H, W)`. Batch dim and channel dim must both be 1.
        All tensors must have the same number of dimensions.
    :param bool screenshot: if ``True``, capture a screenshot after rendering,
        close the viewer, and return a `PIL.Image.Image`.
    :return: ``None``, or a `PIL.Image.Image` if ``screenshot`` is ``True``.
    """ 
    try:
        import napari
        from PIL import Image
    except ImportError:
        raise ImportError(
            "plot_napari requires napari, which is not installed. "
            'Please install it with `pip install "napari[all]"`.'
        )

    n = len(x)
    if n == 0 or n > 6:
        raise ValueError(f"Expected 1-6 tensors, got {n}")
    if not all(isinstance(a, Tensor) for a in x):
        raise TypeError("All inputs must be torch.Tensor")

    ndims = {a.ndim for a in x}
    if ndims not in ({4}, {5}):
        raise ValueError(
            f"All inputs must be either all 4D (1,1,H,W) or all 5D (1,1,D,H,W), got ndims {ndims}"
        )
    is_3d = 5 in ndims

    for a in x:
        if a.shape[0] != 1 or a.shape[1] != 1:
            raise ValueError(
                f"Expected batch dim 1 and channel dim 1, got shape {tuple(a.shape)}"
            )

    arrays = [a.squeeze(1).squeeze(0).detach().cpu().numpy() for a in x]
    arrays = [arr / arr.max() for arr in arrays]

    viewer = napari.Viewer(ndisplay=3 if is_3d else 2, show=True)

    for i, arr in enumerate(arrays):
        viewer.add_image(arr, name=f"{i}")

    if n > 1:
        viewer.grid.enabled = True
        if n == 2:
            viewer.grid.shape = (1, 2)
        elif n == 3:
            viewer.grid.shape = (1, 3)
        elif n == 4:
            viewer.grid.shape = (2, 2)
        elif n in (5, 6):
            viewer.grid.shape = (2, 3)

    viewer.reset_view()

    if screenshot:
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()
        img = viewer.screenshot(canvas_only=False)
        viewer.close()
        return Image.fromarray(img)

    napari.run()
