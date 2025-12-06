from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any

from tqdm import tqdm
import os
from warnings import warn

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = ImportError(
        "The h5py package is not installed. Please install it with `pip install h5py`."
    )  # pragma: no cover
import torch
import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader, Subset, Dataset
from deepinv.utils.tensorlist import TensorList
from deepinv.physics import StackedPhysics
from deepinv.datasets.base import ImageDataset

if TYPE_CHECKING:
    from deepinv.physics import Physics
    from deepinv.physics.generator import PhysicsGenerator
    from deepinv.transform import Transform


class HDF5Dataset(ImageDataset):
    r"""
    DeepInverse HDF5 dataset with signal/measurement pairs ``(x, y)``.

    If there is no training ground truth (i.e. ``x_train``) in the dataset file,
    the dataset returns the measurement again as the signal.

    Optionally also return physics generator params as a dict per sample ``(x, y, params)``, if one was used during data generation.

    .. note::

        We support all dtypes supported by ``h5py`` including complex numbers, which will be stored as complex dtype.

    :param str path: Path to the folder containing the dataset (one or multiple HDF5 files).
    :param bool train: Set to ``True`` for training and ``False`` for testing. If ``split`` argument used, then ``train`` is ignored.
    :param str split: overrides ``train`` argument if not None. Custom dataset split e.g. "train", "test" or "val", which selects the split name used when generating the dataset.
    :param Transform, Callable transform: A deepinv or torchvision transform to apply to the data.
    :param bool load_physics_generator_params: load physics generator params from dataset if they exist
        (e.g. if dataset created with :func:`deepinv.datasets.generate_dataset`)
    :param torch.dtype, str dtype: cast all real-valued data to this dtype.
    :param torch.dtype, str complex_dtype: cast all complex-valued data to this dtype.
    """

    def __init__(
        self,
        path: str,
        train: bool = True,
        split: str = None,
        transform: Transform | Callable = None,
        load_physics_generator_params: bool = False,
        dtype: torch.dtype = torch.float,
        complex_dtype: torch.dtype = torch.cfloat,
    ):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.unsupervised = False
        self.transform = transform
        self.load_physics_generator_params = load_physics_generator_params
        self.cast = lambda x: x.type(complex_dtype if x.is_complex() else dtype)

        if isinstance(h5py, ImportError):  # pragma: no cover
            raise h5py

        self.hd5 = h5py.File(path, "r")
        suffix = ("_train" if train else "_test") if split is None else f"_{split}"

        if "stacked" in self.hd5.attrs.keys():
            self.stacked = self.hd5.attrs["stacked"]
            self.y = [self.hd5[f"y{i}{suffix}"] for i in range(self.stacked)]
        else:
            self.stacked = 0
            self.y = self.hd5[f"y{suffix}"]

        if train or split == "train":
            if "x_train" in self.hd5:
                self.x = self.hd5[f"x{suffix}"]
            else:
                self.unsupervised = True
        else:
            self.x = self.hd5[f"x{suffix}"]

        if self.load_physics_generator_params:
            self.params = {}
            for k in self.hd5:
                if suffix in k and k not in (f"x{suffix}", f"y{suffix}"):
                    self.params[k.replace(suffix, "")] = self.hd5[k]

    def __getitem__(self, index):
        r"""
        Returns the measurement and signal pair ``(x, y)`` at the given index.

        If there is no training ground truth (i.e. ``x_train``) in the dataset file,
        the dataset returns the measurement again as the signal.

        :param int index: Index of the pair to return.
        """
        if self.hd5 is None:  # pragma: no cover
            raise ValueError(
                "Dataset has been closed. Redefine the dataset to continue."
            )

        if self.stacked > 0:
            y = TensorList([self.cast(torch.from_numpy(y[index])) for y in self.y])
        else:
            y = self.cast(torch.from_numpy(self.y[index]))

        if not self.unsupervised:
            x = self.cast(torch.from_numpy(self.x[index]))

            if self.transform is not None:
                x = self.transform(x)
        else:
            x = torch.tensor(torch.nan, dtype=y.dtype, device=y.device)

        if self.load_physics_generator_params:
            params = {
                k: self.cast(
                    torch.from_numpy(param[index])
                    if param.ndim > 1
                    else torch.tensor(param[index])
                )
                for (k, param) in self.params.items()
            }
            return x, y, params
        else:
            return x, y

    def __len__(self):
        r"""
        Returns the size of the dataset.

        """
        if self.stacked > 0:
            return len(self.y[0])
        else:
            return len(self.y)

    def close(self):
        """
        Closes the HDF5 dataset. Use when you are finished with the dataset.
        """
        if hasattr(self, "hd5") and self.hd5:
            self.hd5.close()
            self.hd5 = None


def collate(dataset: Dataset) -> Callable[[list[Any]], Tensor] | None:
    """
    Infer an appropriate `collate_fn` for a DataLoader based on the type returned by the dataset.

    Because `dataset` may return arbitrary Python objects, and we have no control
    over its internals, this helper inspects the first sample and constructs a
    custom `collate_fn` when necessary (for now, this is only implemented for PIL Images)

    Behaviour
    ---------
    - If the dataset returns `torch.Tensor` or `np.ndarray` (or lists/tuples
      whose first element is one of these), `None` is returned. The default
      DataLoader collate logic already handles batching these types.

    - If the dataset returns a PIL Image (or a list/tuple whose first element
      is a PIL Image),  a custom collate function is returned that:
        * converts each image to a float32 tensor in [0, 1] with shape
          `(C, H, W)` using `torchvision.transforms.ToTensor`,
        * stacks all samples into a batched tensor of shape (N, C, H, W), and
        * raises an error if any images differ in shape.

    - Any other return type leads to a `RuntimeError`.

    Only the first sample (`dataset[0]`) is inspected, so the dataset is
    assumed to be type-consistent.
    """
    example_output = dataset[0]
    example_output = (
        example_output[0]
        if isinstance(example_output, (list, tuple))
        else example_output
    )

    if isinstance(example_output, (Tensor, np.ndarray)):
        return None
    else:
        from PIL import Image

        if isinstance(example_output, Image.Image):

            def collate_pillow(
                batch: list[Image.Image | list[Image.Image]],
            ) -> Tensor:
                tensors = []
                for sample in batch:
                    if isinstance(sample, Image.Image):
                        img = sample
                    elif isinstance(sample, (list, tuple)):
                        # only keeping the first element is same behavior as when dataset returns list of tensors!
                        img = sample[0]
                    else:  # pragma: no cover
                        raise ValueError(
                            f"generate_dataset expects datasets to consistently return a (list of) Tensor, Array, or PIL images. Detected use of PIL in a sample, but received a new item of type {type(sample)}."
                        )
                    arr = np.array(img, dtype=np.float32)
                    if arr.ndim == 2:
                        arr = arr[:, :, None]
                    t = torch.from_numpy(arr.transpose(2, 0, 1))
                    t.div_(255.0)
                    if tensors and t.shape != tensors[-1].shape:
                        raise RuntimeError(
                            f"generate_dataset expects dataset to return elements of same shape, but received at two different shapes: {t.shape} and {tensors[-1].shape}. Please add a crop/pad or other shape handling to your dataset."
                        )
                    tensors.append(t)
                return torch.stack(tensors, dim=0)

            return collate_pillow
        else:  # pragma: no cover
            raise RuntimeError(
                f"Dataset must return either numpy array, torch tensor, or PIL image, but got type {type(example_output)}"
            )


def generate_dataset(
    train_dataset: Dataset,
    physics: Physics | list[Physics],
    save_dir: str,
    test_dataset: Dataset = None,
    val_dataset: Dataset = None,
    dataset_filename: str = "dinv_dataset",
    overwrite_existing: bool = True,
    train_datapoints: int = None,
    test_datapoints: int = None,
    val_datapoints: int = None,
    physics_generator: PhysicsGenerator = None,
    save_physics_generator_params: bool = True,
    batch_size: int = 4,
    num_workers: int = 0,
    supervised: bool = True,
    verbose: bool = True,
    show_progress_bar: bool = False,
    device: torch.device | str = "cpu",
) -> str | list[str]:
    r"""
    Generates dataset of signal/measurement pairs from base dataset.

    It generates the measurement data using the forward operator provided by the user.
    The dataset is saved in HDF5 format and can be easily loaded using the :class:`deepinv.datasets.HDF5Dataset` class.
    The generated dataset contains `train` and `test` splits.

    The base dataset of ground-truth images must return tensors `x` or tuples `(x, ...)`. We provide a large library of predefined
    popular imaging datasets. See :ref:`datasets user guide <datasets>` for more information.

    Optionally, if random physics generator is used to generate data, also save physics generator params.
    This is useful e.g. if you are performing a parameter estimation task and want to evaluate the learnt parameters,
    or for measurement consistency/data fidelity, and require knowledge of the params when constructing the loss.

    .. note::

        We support all dtypes supported by ``h5py`` including complex numbers, which will be stored as complex dtype.

    .. note::

        By default, we overwrite existing datasets if they have been previously created. To avoid this, set ``overwrite_existing=False``.

    :param torch.utils.data.Dataset train_dataset: base dataset of ground-truth images. Must return tensors `x` or tuples `(x, ...)`.
    :param deepinv.physics.Physics physics: Forward operator used to generate the measurement data.
        It can be either a single operator or a list of forward operators. In the latter case, the dataset will be
        assigned evenly across operators.
    :param str save_dir: folder where the dataset and forward operator will be saved.
    :param torch.utils.data.Dataset test_dataset: if included, the function will also generate measurements associated to the test dataset.
    :param torch.utils.data.Dataset val_dataset: if included, the function will also generate measurements associated to the validation dataset.
    :param str dataset_filename: desired filename of the dataset (without extension).
    :param bool overwrite_existing: if ``True``, create new dataset file, overwriting any existing dataset with the same ``dataset_filename``.
        If ``False`` and dataset file already exists, does not create new dataset.
    :param int, None train_datapoints: Desired number of datapoints in the training dataset. If set to ``None``, it will use the
        number of datapoints in the base dataset. This is useful for generating a larger train dataset via data
        augmentation (which should be chosen in the train_dataset).
    :param int, None test_datapoints: Desired number of datapoints in the test dataset. If set to ``None``, it will use the
        number of datapoints in the base test dataset.
    :param int, None val_datapoints: Desired number of datapoints in the val dataset.
    :param None, deepinv.physics.generator.PhysicsGenerator physics_generator: Optional physics generator for generating
            the physics operators. If not None, the physics operators are randomly sampled at each iteration using the generator.
    :param bool save_physics_generator_params: save physics generator params too, ignored if ``physics_generator`` not used.
    :param int batch_size: batch size for generating the measurement data
        (it affects the speed of the generating process, and the physics generator batch size)
    :param int num_workers: number of workers for generating the measurement data
        (it only affects the speed of the generating process)
    :param bool supervised: Generates supervised pairs ``(x,y)`` of measurements and signals.
        If set to ``False``, it will generate a training dataset with measurements only ``(y)``
        and a test dataset with pairs ``(x,y)``
    :param bool verbose: Output progress information in the console.
    :param bool show_progress_bar: Show progress bar during the generation
        of the dataset (if verbose is set to ``True``).
    :param torch.device, str device: device, e.g. cpu or gpu, on which to generate measurements. All data is moved back to cpu before saving.

    """
    if isinstance(h5py, ImportError):
        raise h5py

    if test_dataset is None and train_dataset is None and val_dataset is None:
        raise ValueError("No train or test datasets provided.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not isinstance(physics, (list, tuple)):
        physics = [physics]

    physics = [p.clone() for p in physics]

    G = len(physics)

    save_physics_generator_params = (
        save_physics_generator_params and physics_generator is not None
    )

    if train_dataset is not None:
        n_train = train_datapoints or len(train_dataset)
        n_train_g = int(n_train / G)
        n_dataset_g = int(min(len(train_dataset), n_train) / G)

    if test_dataset is not None:
        n_test = min(len(test_dataset), test_datapoints or len(test_dataset))
        n_test_g = int(n_test / G)

    if val_dataset is not None:
        n_val = min(len(val_dataset), val_datapoints or len(val_dataset))
        n_val_g = int(n_val / G)

    hf_paths = []

    for g in range(G):
        hf_path = f"{save_dir}/{dataset_filename}{g}.h5"
        hf_paths.append(hf_path)

        if os.path.exists(hf_path):
            if overwrite_existing:
                warn(
                    f"Dataset {hf_path} already exists, this will close and overwrite the previous dataset."
                )
                # remove existing dataset to avoid open file error
                os.remove(hf_path)
            else:
                warn(f"Dataset {hf_path} already exists, skipping...")
                continue

        hf = h5py.File(hf_path, "w")

        op_g = physics[g]
        hf.attrs["operator"] = op_g.__class__.__name__
        if isinstance(op_g, StackedPhysics):
            hf.attrs["stacked"] = len(op_g)
        if physics_generator is not None:
            physics_generator.reset_rng()

        def measure(x: Tensor, b: int) -> tuple[Tensor | TensorList, dict | None]:
            if physics_generator is None:
                return op_g(x), None
            params = physics_generator.step(batch_size=b)
            return op_g(x, **params), params

        torch.save(op_g.state_dict(), f"{save_dir}/physics{g}.pt")

        created_splits: set[str] = set()

        def process_batch(
            hf_ds: HDF5Dataset,
            x_batch,
            split_name: str,
            index: int,
            n_split: int,
        ) -> int:
            """Process one batch for a given split and return updated index."""
            x = x_batch[0] if isinstance(x_batch, (list, tuple)) else x_batch
            x = x.to(device)

            bsize = x.size(0)
            if split_name == "train" and index + bsize > n_split:
                bsize = n_split - index
            if bsize <= 0:
                return index
            y, params = measure(x, b=bsize)

            if split_name not in created_splits:
                if isinstance(y, TensorList):
                    for i in range(len(y)):
                        hf_ds.create_dataset(
                            f"y{i}_{split_name}",
                            (n_split,) + y[i].shape[1:],
                            dtype=y[0].cpu().numpy().dtype,
                        )
                else:
                    hf_ds.create_dataset(
                        f"y_{split_name}",
                        (n_split,) + y.shape[1:],
                        dtype=y.cpu().numpy().dtype,
                    )

                if split_name != "train" or supervised:
                    hf_ds.create_dataset(
                        f"x_{split_name}",
                        (n_split,) + x.shape[1:],
                        dtype=x.cpu().numpy().dtype,
                    )
                if save_physics_generator_params and params is not None:
                    for k, p in params.items():
                        hf_ds.create_dataset(
                            f"{k}_{split_name}",
                            (n_split,) + p.shape[1:],
                            dtype=p.cpu().numpy().dtype,
                        )
                created_splits.add(split_name)

            sl = slice(index, index + bsize)

            if isinstance(y, TensorList):
                for i in range(len(y)):
                    hf_ds[f"y{i}_{split_name}"][sl] = y[i][:bsize].cpu().numpy()
            else:
                hf_ds[f"y_{split_name}"][sl] = y[:bsize].cpu().numpy()

            if split_name != "train" or supervised:
                hf_ds[f"x_{split_name}"][sl] = x[:bsize].cpu().numpy()

            if save_physics_generator_params and params is not None:
                for k, p in params.items():
                    hf_ds[f"{k}_{split_name}"][sl] = p[:bsize].cpu().numpy()

            return index + bsize

        split_specs = []
        if train_dataset is not None:
            epochs = int(n_train_g / len(train_dataset)) + 1
            train_indices = list(range(g * n_dataset_g, (g + 1) * n_dataset_g))
            split_specs.append(
                ("train", train_dataset, n_train_g, train_indices, epochs)
            )

        if test_dataset is not None:
            test_indices = list(range(g * n_test_g, (g + 1) * n_test_g))
            split_specs.append(("test", test_dataset, n_test_g, test_indices, 1))

        if val_dataset is not None:
            val_indices = list(range(g * n_val_g, (g + 1) * n_val_g))
            split_specs.append(("val", val_dataset, n_val_g, val_indices, 1))

        for split_name, dataset, n_split, indices, epochs in split_specs:
            if n_split <= 0:
                continue

            subset = Subset(dataset, indices=indices)

            epoch_iter = range(epochs)
            if split_name == "train":
                epoch_iter = tqdm(
                    epoch_iter,
                    ncols=150,
                    disable=(not verbose or not show_progress_bar),
                )

            index = 0
            for e in epoch_iter:
                if split_name == "train":
                    desc = (
                        f"Generating dataset operator {g + 1}"
                        if G > 1
                        else f"Generating {split_name} dataset"
                    )
                    epoch_iter.set_description(desc)

                dataloader = DataLoader(
                    subset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=torch.device(device).type != "cpu",
                    drop_last=False,
                    collate_fn=collate(dataset),
                )

                for x_batch in dataloader:
                    index = process_batch(
                        hf,
                        x_batch,
                        split_name,
                        index,
                        n_split,
                    )

                    # for train, once we've filled n_split samples, we stop
                    if split_name == "train" and index >= n_split:
                        break

                if split_name == "train" and index >= n_split:
                    break

        hf.close()

        if verbose:
            print(f"Dataset has been saved at {hf_path}")

    return hf_paths[0] if G == 1 else hf_paths
