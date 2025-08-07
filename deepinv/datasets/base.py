from warnings import warn
from typing import Optional, Callable, Union
from pathlib import Path
import math
from natsort import natsorted
from numpy import ndarray
import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image
from PIL.Image import Image as PIL_Image
from deepinv.utils.tensorlist import TensorList

CORE_TYPES = (
    Tensor,
    TensorList,
)


def check_dataset(dataset: Dataset, allow_non_tensor=True) -> None:
    """Check that a torch dataset is compatible with DeepInverse.

    For details of what is compatible, see :class:`ImageDataset`.

    :param torch.utils.data.Dataset dataset: torch dataset.
    :param bool allow_non_tensor: allow image types that are not tensors (i.e. numpy ndarrays and PIL Images). Default `False`, which
        is recommended so that the dataset is asserted to return tensors to be compatible with deepinv.
    """

    image_types = CORE_TYPES + ((PIL_Image, ndarray, float) if allow_non_tensor else ())

    if len(dataset) <= 0:
        raise RuntimeError(f"Dataset {dataset} should have length greater than zero.")

    batch = dataset.__getitem__(0)
    error = f"Dataset {dataset} should return either non-nan image `x`, or tuples of either length 2 of 3 of (x, y) or (x, params), or (x, y, params), where x, y are images (either Tensor, TensorList, or nan) and params is a dict"

    def warn_core_types(x):
        if (
            isinstance(x, image_types)
            and not isinstance(x, CORE_TYPES)
            and not (isinstance(x, float) and math.isnan(x))
        ):
            warn(
                f"Dataset is returning samples of type {type(x)} that are not Tensors or TensorLists. These are not natively supported by DeepInverse. Pass in a transform to your dataset to cast them to Tensors before using the dataset with DeepInverse."
            )

    if isinstance(batch, image_types):
        warn_core_types(batch)
        if isinstance(batch, CORE_TYPES) and batch.isnan().all():
            raise RuntimeError(f"{error}, but returned all nan tensor `x`.")
        elif isinstance(batch, float) and math.isnan(batch):
            raise RuntimeError(f"{error}, but returned {batch}.")

    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y_or_params = batch

        warn_core_types(x)
        if not isinstance(x, image_types):
            raise RuntimeError(
                f"{error}, but index 0 of returned tuple is type {type(x)}."
            )

        warn_core_types(y_or_params)
        if not isinstance(y_or_params, (*image_types, dict)):
            raise RuntimeError(
                f"{error}, but index 1 of returned tuple is type {type(y_or_params)}."
            )
        elif isinstance(y_or_params, dict) and any(
            not isinstance(k, str) for k in y_or_params
        ):
            raise RuntimeError(f"{error}, but params dict has non-string keys.")

    elif isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, y, params = batch

        warn_core_types(x)
        if not isinstance(x, image_types):
            raise RuntimeError(
                f"{error}, but index 0 of returned tuple is type {type(x)}."
            )

        warn_core_types(y)
        if not isinstance(y, image_types):
            raise RuntimeError(
                f"{error}, but index 1 of returned tuple is type {type(y)}."
            )

        if not isinstance(params, dict):
            raise RuntimeError(
                f"{error}, but index 2 of returned tuple is not dict but of type {type(params)}."
            )
        elif any(not isinstance(k, str) for k in params):
            raise RuntimeError(f"{error}, but params dict has non-string keys.")

    elif isinstance(batch, (list, tuple)):
        raise RuntimeError(
            f"{error}, but returned list or tuple of length {len(batch)}."
        )

    else:
        raise RuntimeError(f"{error}, but returned batch of type {type(batch)}.")


class ImageDataset(Dataset):
    """
    Base class for imaging datasets in DeepInverse.

    All datasets used with DeepInverse should inherit from this class.

    We provide the function :func:`check_dataset` to automatically check that `__getitem__` returns the correct format out of the following options:

    * `x` i.e a dataset that returns only ground truth;
    * `(x, y)` i.e. a dataset that returns pairs of ground truth and measurement. `x` can be equal to `torch.nan` if your dataset is ground-truth-free.
    * `(x, params)` i.e. a dataset of ground truth and dict of :ref:`physics parameters <physics_generators>`. Useful for training with online measurements.
    * `(x, y, params)` i.e. a dataset that returns ground truth, measurements and dict of physics params.

    This check is also available for datasets using the method :meth:`ImageDataset.check_dataset`.

    .. tip:

        If you have a dataset of measurements only `(y)` or `(y, params)` you should modify it such that it returns `(torch.nan, y)` or `(torch.nan, y, params)`

    Datasets should ideally return :class:`torch.Tensor` or :class:`deepinv.utils.TensorList` so that they are batchable and can be used with `deepinv`.

    If using DeepInverse with your own custom dataset, you should inherit from this class and use :func:`check_dataset` to check your dataset is compatible.
    """

    def check_dataset(self):
        """Check dataset returns correct format of images or image tuples."""
        check_dataset(self, allow_non_tensor=True)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx: int):
        raise NotImplementedError()


class TensorDataset(ImageDataset):
    r"""
    Dataset wrapping data explicitly passed as tensors.

    This dataset can be used to return ground truth `x`, ground truth and measurements `(x, y)`, or measurements only `(y)`.
    All input tensors must be of shape `(N, ...)` and of same `N` where N is the number of samples and ... represents the data dimensions.

    Optionally, `params` are returned too.

    :param torch.Tensor, None x: optional input ground truth tensor `x`
    :param torch.Tensor, None y: optional input measurement tensor `y`
    :param dict[str, torch.Tensor], None params: optional input physics parameters `params` of format `{"str": Tensor}`

    |sep|

    Examples:

    Construct a dataset from a single measurement only:

    >>> import torch
    >>> from deepinv.datasets import TensorDataset
    >>> y = torch.rand(1, 3, 8, 8) # B,C,H,W
    >>> dataset = TensorDataset(y=y)
    >>> x, y = dataset[0]
    >>> x
    nan
    >>> y.shape
    torch.Size([3, 8, 8])

    Construct a dataset from a ground truth batch:

    >>> x = torch.rand(4, 3, 8, 8)  # 4 samples of 3-channel 8x8 images
    >>> dataset = TensorDataset(x=x)
    >>> dataset[0].shape
    torch.Size([3, 8, 8])


    """

    def __init__(
        self,
        *,
        x: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        params: Optional[dict[str, Tensor]] = None,
    ):
        super().__init__()

        if (
            isinstance(x, CORE_TYPES)
            and isinstance(y, CORE_TYPES)
            and x.size(0) != y.size(0)
        ):
            raise ValueError(
                f"x must be same size as y in dimension 0, but got {x.size(0)} vs {y.size(0)}"
            )
        elif self._is_none_or_nan(x) and self._is_none_or_nan(y):
            raise ValueError("At least one of x or y must be not None or not nan.")
        elif not self._is_none_or_nan(x) and not isinstance(x, CORE_TYPES):
            raise ValueError("x must be Tensor or TensorList.")
        elif not self._is_none_or_nan(y) and not isinstance(y, CORE_TYPES):
            raise ValueError("y must be Tensor or TensorList.")

        self._x = x
        self._y = y
        self._params = params

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def params(self):
        return self._params

    @staticmethod
    def _is_none_or_nan(x) -> bool:
        """
        Check if x is None or is nan
        """
        return x is None or (isinstance(x, float) and math.isnan(x))

    def __len__(self):
        return self.x.size(0) if not self._is_none_or_nan(self.x) else self.y.size(0)

    def __getitem__(self, idx: int):
        if self._is_none_or_nan(self.y):
            if self._is_none_or_nan(self.params):
                return self.x[idx]
            else:
                return self.x[idx], {k: v[idx] for (k, v) in self.params.items()}

        x = torch.nan if self._is_none_or_nan(self.x) else self.x[idx]

        if self._is_none_or_nan(self.params):
            return x, self.y[idx]
        else:
            return x, self.y[idx], {k: v[idx] for (k, v) in self.params.items()}


class ImageFolder(ImageDataset):
    """Dataset loading images from files.

    By default, the images are loaded from image files (png, jpg etc.) located in `root`.

    For more flexibility, set `x_path` or `y_path` to load ground truth `x` and/or measurements `y` from specific file patterns.

    .. tip::

        To load data from subfolders, use globs such as `x_path = "GT/**/*.png", y_path = "meas/**/*.png"`.

    .. tip::

        Set `y_path` only to load measurements following the file pattern. The measurement-only data will be returned as a tuple `(torch.nan, y)`.

    :param str, pathlib.Path root: dataset root directory.
    :param str, None x_path: file glob pattern for ground truth data, defaults to None.
    :param str, None y_path: file glob pattern for measurement data, defaults to None.
    :param Callable loader: optional function that takes filename string and loads file. If `None`, defaults to `PIL.Image.open`.
    :param Callable estimate_params: optional function that takes tensors `x,y` and returns dict of `params`. Advanced usage only.
    :param Callable, tuple transform: optional callable transform. If `tuple` or `list` of length 2, `x` is transformed with first transform and `y` with second.

    |sep|

    Examples:

    Using default loading from root folder with image files. Folder structure:

    ::

        root
        ├── img1.png
        └── img2.png

        dataset = ImageFolder(root)
        dataset[0]
        tensor(...)  # Returns x only

    Loading paired tensors from nested folders using custom glob and loader. Folder structure:

    ::

        data/
        ├── GT/
        │   ├── scene1/
        │   │   └── x0.pt
        │   └── scene2/
        │       └── x1.pt
        └── meas/
            ├── scene1/
            │   └── y0.pt
            └── scene2/
                └── y1.pt

        dataset = ImageFolder(
            root,
            x_path="GT/**/*.pt",
            y_path="meas/**/*.pt",
            loader=torch.load
        )
        dataset[0]
        (tensor(...), tensor(...))  # Returns (x, y) pair


    Loading unpaired measurements only. Folder structure:

    ::

        data/
        └── meas/
            ├── meas0.png
            └── meas1.png

        dataset = ImageFolder(
            "data/",
            y_path="meas/*.png"
        )
        dataset[0]
        (torch.nan, tensor(...))  # Returns unpaired y

    """

    def __init__(
        self,
        root: Union[str, Path],
        x_path: Optional[str] = None,
        y_path: Optional[str] = None,
        loader: Callable[[Union[str, Path]], Tensor] = None,
        estimate_params: Optional[Callable[[Tensor, Tensor], dict]] = None,
        transform: Optional[Union[Callable, tuple[Callable, Callable]]] = None,
    ):
        super().__init__()
        self.root = Path(root)

        self.x_paths = None
        self.y_paths = None

        if x_path is not None:
            self.x_paths = natsorted(self.root.glob(x_path))

        if y_path is not None:
            self.y_paths = natsorted(self.root.glob(y_path))

        if (
            self.x_paths is not None
            and self.y_paths is not None
            and len(self.x_paths) != len(self.y_paths)
        ):
            raise ValueError("Mismatch in number of GT and LR images.")
        elif self.x_paths is None and self.y_paths is None:
            self.x_paths = sum(
                (list(self.root.glob(f"**/*{ext}")) for ext in IMG_EXTENSIONS), []
            )

        if transform is None:
            self.transform_x = self.transform_y = lambda x: x
        elif isinstance(transform, (list, tuple)):
            if len(transform) != 2:
                raise ValueError(
                    "If transform is iterable, it should be length 2 (for x and y)."
                )
            self.transform_x, self.transform_y = transform
        else:
            self.transform_x = self.transform_y = transform

        self.estimate_params = estimate_params
        self.loader = (
            loader if loader is not None else lambda fn: Image.open(fn).convert("RGB")
        )

    def __len__(self):
        return len(self.x_paths) if self.x_paths is not None else len(self.y_paths)

    def __getitem__(self, idx):
        if self.x_paths is None:
            x = torch.nan
        else:
            x = self.transform_x(self.loader(self.x_paths[idx]))

        if self.y_paths is None:
            y = None
        else:
            y = self.transform_y(self.loader(self.y_paths[idx]))

        params = self.estimate_params(x, y) if self.estimate_params is not None else {}

        out = (x,)

        if y is not None:
            out += (y,)

        if params:
            out += (params,)

        return out[0] if len(out) == 1 else out
