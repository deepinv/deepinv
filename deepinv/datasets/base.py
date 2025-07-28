from functools import wraps
from torch.utils.data import Dataset
from torch import Tensor
from PIL.Image import Image


def check_dataset(dataset: Dataset) -> None:
    """Check that a torch dataset is compatible with DeepInverse.

    For details of what is compatible, see :class:`ImagingDataset`.

    :param torch.utils.data.Dataset dataset: torch dataset.
    """
    if len(dataset) <= 0:
        raise RuntimeError(f"Dataset {dataset} should have length greater than zero.")

    batch = dataset.__getitem__(0)
    error = f"Dataset {dataset} should return either non-nan image `x`, or tuples of either length 2 of 3 of (x, y) or (x, params), or (x, y, params), where x, y are images and params is a dict"

    if isinstance(batch, (Tensor, Image)):
        if isinstance(batch, Tensor) and batch.isnan().all():
            raise RuntimeError(f"{error}, but returned all nan tensor `x`.")
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y_or_params = batch
        if not isinstance(x, (Tensor, Image)):
            raise RuntimeError(
                f"{error}, but index 0 of returned tuple is not Tensor or Image."
            )
        if not isinstance(y_or_params, (Tensor, Image)) and not isinstance(
            y_or_params, dict
        ):
            raise RuntimeError(
                f"{error}, but index 1 of returned tuple is not Tensor nor dict."
            )
    elif isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, y, params = batch
        if not isinstance(x, (Tensor, Image)):
            raise RuntimeError(
                f"{error}, but index 0 of returned tuple is not Tensor or Image."
            )
        if not isinstance(y, (Tensor, Image)):
            raise RuntimeError(
                f"{error}, but index 1 of returned tuple is not Tensor or Image."
            )
        if isinstance(params, dict):
            raise RuntimeError(f"{error}, but index 2 of returned tuple is not dict.")
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

    *

    If using DeepInverse with your own custom dataset, it should either inherit from this class,
    or use the :func:`check_dataset` function to check your dataset is compatible.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        init = cls.__init__

        @wraps(init)
        def new_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            check_dataset(self)

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
