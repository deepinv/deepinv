from functools import wraps
from warnings import warn
from numpy import ndarray
from torch.utils.data import Dataset
from torch import Tensor
from PIL.Image import Image
from deepinv.utils.tensorlist import TensorList


def check_dataset(dataset: Dataset, allow_non_tensor=True) -> None:
    """Check that a torch dataset is compatible with DeepInverse.

    For details of what is compatible, see :class:`BaseDataset`.

    :param torch.utils.data.Dataset dataset: torch dataset.
    :param bool allow_non_tensor: allow image types that are not tensors (i.e. numpy ndarrays and PIL Images). Default `False`, which
        is recommended so that the dataset is asserted to return tensors to be compatible with deepinv.
    """
    core_types = (Tensor, TensorList)
    image_types = core_types + ((Image, ndarray) if allow_non_tensor else ())

    if len(dataset) <= 0:
        raise RuntimeError(f"Dataset {dataset} should have length greater than zero.")

    batch = dataset.__getitem__(0)
    error = f"Dataset {dataset} should return either non-nan image `x`, or tuples of either length 2 of 3 of (x, y) or (x, params), or (x, y, params), where x, y are images (Tensor, TensorList) and params is a dict"

    def warn_core_types(x):
        if isinstance(x, image_types) and not isinstance(x, core_types):
            warn(
                f"Dataset is returning samples of type {type(x)} that are not Tensors or TensorLists. These are not natively supported by DeepInverse. Pass in a transform to your dataset to cast them to Tensors before using the dataset with DeepInverse."
            )

    if isinstance(batch, image_types):
        if isinstance(batch, core_types) and batch.isnan().all():
            raise RuntimeError(f"{error}, but returned all nan tensor `x`.")
        warn_core_types(batch)

    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y_or_params = batch

        warn_core_types(x)
        if not isinstance(x, image_types):
            raise RuntimeError(
                f"{error}, but index 0 of returned tuple is type {type(batch)}."
            )

        warn_core_types(y_or_params)
        if not isinstance(y_or_params, (*image_types, dict)):
            raise RuntimeError(
                f"{error}, but index 1 of returned tuple is type {type(batch)}."
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
                f"{error}, but index 0 of returned tuple is type {type(batch)}."
            )

        warn_core_types(y)
        if not isinstance(y, image_types):
            raise RuntimeError(
                f"{error}, but index 1 of returned tuple is type {type(batch)}."
            )

        if not isinstance(params, dict):
            raise RuntimeError(f"{error}, but index 2 of returned tuple is not dict.")
        elif any(not isinstance(k, str) for k in params):
            raise RuntimeError(f"{error}, but params dict has non-string keys.")

    elif isinstance(batch, (list, tuple)):
        raise RuntimeError(
            f"{error}, but returned list or tuple of length {len(batch)}."
        )

    else:
        raise RuntimeError(f"{error}, but returned batch of type {type(batch)}.")


class BaseDataset(Dataset):
    """
    Base class for imaging datasets in DeepInverse.

    All datasets used with DeepInverse should inherit from this class. The dataset uses :func:`check_dataset` to
    automatically check that `__getitem__` returns the correct format:

    Assuming that `x` is the ground-truth reference and `y` is the measurement and `params` is a dict of :ref:`physics parameters <physics_generators>`,
    the **dataloaders** should return one of the following options:

    1. `(x, y)` or `(x, y, params)`, which requires `online_measurements=False` (default) otherwise `y` will be ignored and new measurements will be generated online.
    2. `(x)` or `(x, params)`, which requires `online_measurements=True` for generating measurements in an online manner (optionally with parameters) as `y=physics(x)` or `y=physics(x, **params)`. Otherwise, first generate a dataset of `(x,y)` with :class:`deepinv.datasets.generate_dataset` and then use option 1 above.
    3. If you have a dataset of measurements only `(y)` or `(y, params)` you should modify it such that it returns `(torch.nan, y)` or `(torch.nan, y, params)`. Set `online_measurements=False`.

    TODO data types must be Tensor or TensorList so that they are batchable and can be used with deepinv.

    If using DeepInverse with your own custom dataset, it should either inherit from this class,
    or use the :func:`check_dataset` function to check your dataset is compatible.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        init = cls.__init__

        @wraps(init)
        def new_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            check_dataset(self, allow_non_tensor=True)

        cls.__init__ = new_init

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx: int):
        raise NotImplementedError()


if __name__ == "__main__":

    class MyDataset(BaseDataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    x = Tensor([1, 2, 3])

    # This should be fine
    dataset = MyDataset(
        [
            [x, x],
        ]
    )

    # This should raise error
    dataset = MyDataset(
        [
            [x, x, x, x],
        ]
    )
