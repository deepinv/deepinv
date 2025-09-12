from __future__ import annotations
from typing import TYPE_CHECKING, Union, Callable

from tqdm import tqdm
import os
from warnings import warn
import re

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, Dataset
from deepinv.utils.tensorlist import TensorList
from deepinv.physics import StackedPhysics
from deepinv.datasets.base import ImageDataset

if TYPE_CHECKING:
    from deepinv.physics import Physics
    from deepinv.physics.generator import PhysicsGenerator
    from deepinv.transform import Transform
    from typing import Any, Optional


def _register_deprecated_attr(
    self: Any,
    *,
    attr_name: str,
    attr_underscore_name: str,
    attr_initial_value: Any,
    deprecation_message: str,
    doc: Optional[str] = None,
) -> None:
    """Deprecate an instance attribute.

    It wraps the attribute so that a warning is raised any time the attribute is read, written, or deleted.

    :param self: The instance to which the attribute is added.
    :param str attr_name: The name of the attribute to be deprecated.
    :param str attr_underscore_name: The name of the internal attribute to store the value.
    :param Any attr_initial_value: The initial value of the attribute.
    :param str deprecation_message: The deprecation warning message to be shown.
    :param str, None doc: The docstring for the deprecated attribute.
    """
    setattr(self, attr_underscore_name, attr_initial_value)

    def fget(self) -> bool:
        val = getattr(self, attr_underscore_name)
        # warn last in case retrieval fails
        warn(deprecation_message, DeprecationWarning, sacklevel=2)
        return val

    def fset(self, value: bool) -> None:
        setattr(self, attr_underscore_name, value)
        # warn last in case setting fails
        warn(deprecation_message, DeprecationWarning, sacklevel=2)

    def fdel(self) -> None:
        delattr(self, attr_underscore_name)
        # warn last in case deletion fails
        warn(deprecation_message, DeprecationWarning, sacklevel=2)

    attr_value = property(fget=fget, fset=fset, fdel=fdel, doc=doc)
    setattr(self, attr_name, attr_value)


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
        transform: Union[Transform, Callable] = None,
        load_physics_generator_params: bool = False,
        dtype: torch.dtype = torch.float,
        complex_dtype: torch.dtype = torch.cfloat,
    ):
        import h5py

        super().__init__()

        f = h5py.File(path, "r")

        self._register_members(
            f,
            train=train,
            split=split,
            load_physics_generator_params=load_physics_generator_params,
        )

        self.hd5 = f
        self.transform = transform
        self.cast = lambda x: x.type(complex_dtype if x.is_complex() else dtype)

        self._register_deprecated_attributes()

    def _register_members(
        self,
        f: Any,
        *,
        train: bool,
        split: Optional[str],
        load_physics_generator_params: bool,
    ) -> None:
        # Process ground truths
        x = None

        # Process measurements
        attrs: dict = f.attrs
        marked_stacked = "stacked" in attrs
        stacked = attrs.get("stacked", 0)
        if marked_stacked:
            y = [None] * stacked
        else:
            y = None

        # Process forward operator parameters
        if load_physics_generator_params:
            params = {}
        else:
            params = None

        # Register members in the HDF5 file as ground truths, parts of stacked
        # measurements or regular measurements and forward operator parameters
        split_name = split if split is not None else ("train" if train else "test")
        split_suffix = f"_{split_name}"
        for member_name, member in f.items():
            # Only register members corresponding to the selected split
            if member_name.endswith(split_suffix):
                # Get the attribute name by stripping the split suffix
                attr_name = member_name[: -len(split_suffix)]
                # Register the ground truths
                if attr_name == "x":
                    x = member
                # Register the measurements
                elif attr_name == "y":
                    if not marked_stacked:
                        y = member
                    else:
                        warn(
                            f"Dataset marked as stacked but found unstacked member {member_name}. There is probably an error with the dataset.",
                            UserWarning,
                            stacklevel=2,
                        )
                # Register parts of the stacked measurements and forward operator parameters
                else:
                    # Register part of the stacked measurements
                    # Example target values for member_name: "y0_train", "y1_train", ...
                    if attr_name.startswith("y"):
                        stacking_needle = attr_name[1:]
                        # If needle is the canonical base 10 representation of a non-negative integer
                        if (
                            re.fullmatch(r"0|[1-9]+\d+", stacking_needle, re.ASCII)
                            is not None
                        ):
                            stacking_index = int(stacking_needle)
                            # If the stacking index is in the target range
                            if stacking_index in range(stacked):
                                if marked_stacked:
                                    y[stacking_index] = member
                                else:
                                    warn(
                                        f"Dataset not marked as stacked but found stacked member {member_name}. There is probably an error with the dataset.",
                                        UserWarning,
                                        stacklevel=2,
                                    )
                            else:
                                warn(
                                    f"Found stacked measurement member {member_name} with stacking index {stacking_index} outside of the expected range [0, {stacked}). There is probably an error with the dataset.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                        else:
                            warn(
                                f"Found stacked measurement member {member_name} with middle part {stacking_needle} not a non-negative integer represented canonically in base 10. There is probably an error with the dataset.",
                                UserWarning,
                                stacklevel=2,
                            )

                    # Register the forward operator parameters
                    if params is not None:
                        params[attr_name] = member

        # Process ground truths
        if x is not None:
            self.x = x

        # Process measurements
        if marked_stacked:
            if None in y:
                raise ValueError(
                    f"Dataset marked as stacked with {stacked} parts but some parts are missing."
                )
        self.y = y

        # Process forward operator parameters
        if params is not None:
            self.params = params

    def __getitem__(self, index: int) -> tuple:
        r"""Get an entry in the dataset.

        Return the measuremet and signal pair ``(x, y)`` at the given index, in
        the selected split. If forward operator parameters are available, it
        returns ``(x, y, params)`` where ``params`` is a dict of parameters.

        If the split contains no ground truth, it returns a scalar NaN tensor
        as the ground truth, similarly to :class:`deepinv.training.Trainer`.

        :param int index: Index of the pair to return.
        """
        import h5py

        if self.hd5 is None:
            raise ValueError(
                "Dataset has been closed. Redefine the dataset to continue."
            )

        # Compute x
        if hasattr(self, "x"):
            x = self.x[index]
            x = torch.from_numpy(x)
            x = self.cast(x)

            if self.transform is not None:
                x = self.transform(x)
        else:
            x = torch.tensor(torch.nan, dtype=torch.float32, device=torch.device("cpu"))
            x = self.cast(x)

        # Compute y
        y = self.y
        if isinstance(y, h5py.Dataset):
            y = self.y[index]
            y = torch.from_numpy(y)
            y = self.cast(y)
        else:
            y = TensorList([self.cast(torch.from_numpy(yk[index])) for yk in y])

        # Compute params
        if hasattr(self, "params"):
            params = {
                k: self.cast(
                    torch.from_numpy(param[index])
                    if param.ndim > 1
                    else torch.tensor(param[index])
                )
                for (k, param) in self.params.items()
            }
        else:
            params = None

        if params is not None:
            return x, y, params
        else:
            return x, y

    def __len__(self) -> int:
        r"""
        Returns the size of the dataset.

        """
        import h5py

        y = self.y
        if not isinstance(y, h5py.Dataset):
            y = y[0]
        return len(y)

    def close(self) -> None:
        """
        Closes the HDF5 dataset. Use when you are finished with the dataset.
        """
        if hasattr(self, "hd5") and self.hd5:
            self.hd5.close()
            self.hd5 = None

    @property
    def unsupervised(self) -> bool:
        """Test if the split is unsupervised (i.e. contains no ground truths)."""
        return hasattr(self, "x")

    def _register_deprecated_attributes(self) -> None:
        import h5py

        # The attribute load_physics_generator_params is redundant with the attribute params.
        # Indeed, it is true if and only if the attribute params exists.
        _register_deprecated_attr(
            self,
            attr_name="load_physics_generator_params",
            attr_underscore_name="_load_physics_generator_params",
            attr_initial_value=hasattr(self, "params"),
            deprecation_message="The attribute 'load_physics_generator_params' is deprecated and will be removed in future versions. Use the attribute 'params' instead.",
        )

        # The attribute stacked is redundant with the attribute y. It is the
        # number of elements in y if y is a list, otherwise y is a h5py.Dataset
        # and it is 0.
        _register_deprecated_attr(
            self,
            attr_name="stacked",
            attr_underscore_name="_stacked",
            attr_initial_value=0 if isinstance(self.y, h5py.Dataset) else len(self.y),
            deprecation_message="The attribute 'stacked' is deprecated and will be removed in future versions. Use the attribute 'y' instead.",
        )

        # The attribute data_info is used nowhere.
        _register_deprecated_attr(
            self,
            attr_name="data_info",
            attr_underscore_name="_data_info",
            attr_initial_value=[],
            deprecation_message="The attribute 'data_info' is deprecated and will be removed in future versions.",
        )

        # The attribute data_cache is used nowhere.
        _register_deprecated_attr(
            self,
            attr_name="data_cache",
            attr_underscore_name="_data_cache",
            attr_initial_value={},
            deprecation_message="The attribute 'data_cache' is deprecated and will be removed in future versions.",
        )


def generate_dataset(
    train_dataset: Dataset,
    physics: Physics,
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
    device: Union[torch.device, str] = "cpu",
) -> Union[str, list[str]]:
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
    import h5py

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
                    f"Dataset {hf_path} already exists, this will overwrite the previous dataset."
                )
            else:
                warn(f"Dataset {hf_path} already exists, skipping...")
                continue

        hf = h5py.File(hf_path, "w")

        hf.attrs["operator"] = physics[g].__class__.__name__
        if isinstance(physics[g], StackedPhysics):
            hf.attrs["stacked"] = len(physics[g])

        # get initial image for image size
        if train_dataset is not None:
            x0 = train_dataset[0]
        elif test_dataset is not None:
            x0 = test_dataset[0]
        elif val_dataset is not None:
            x0 = val_dataset[0]

        x0 = x0[0] if isinstance(x0, (list, tuple)) else x0
        x0 = x0.to(device).unsqueeze(0)

        # get initial measurement for initial image
        def measure(x: Tensor, b: int, g: int) -> tuple[Tensor, Union[dict, None]]:
            if physics_generator is None:
                return physics[g](x), None
            else:
                params = physics_generator.step(batch_size=b)
                return physics[g](x, **params), params

        y0, params0 = measure(x0, b=1, g=g)

        if physics_generator is not None:
            # Reset physics generator so it starts again at initial rng
            physics_generator.reset_rng()

        # save physics
        torch.save(physics[g].state_dict(), f"{save_dir}/physics{g}.pt")

        if train_dataset is not None:
            if isinstance(y0, TensorList):
                for i in range(len(y0)):
                    hf.create_dataset(
                        f"y{i}_train",
                        (n_train_g,) + y0[i].shape[1:],
                        dtype=y0[0].cpu().numpy().dtype,
                    )
            else:
                hf.create_dataset(
                    "y_train", (n_train_g,) + y0.shape[1:], dtype=y0.cpu().numpy().dtype
                )
            if supervised:
                hf.create_dataset(
                    "x_train", (n_train_g,) + x0.shape[1:], dtype=x0.cpu().numpy().dtype
                )
            if save_physics_generator_params:
                for k, p in params0.items():
                    hf.create_dataset(
                        f"{k}_train",
                        (n_train_g,) + p.shape[1:],
                        dtype=p.cpu().numpy().dtype,
                    )

            index = 0

            epochs = int(n_train_g / len(train_dataset)) + 1
            for e in (
                progress_bar := tqdm(
                    range(epochs),
                    ncols=150,
                    disable=(not verbose or not show_progress_bar),
                )
            ):
                desc = (
                    f"Generating dataset operator {g + 1}"
                    if G > 1
                    else "Generating train dataset"
                )
                progress_bar.set_description(desc)

                train_dataloader = DataLoader(
                    Subset(
                        train_dataset,
                        indices=list(range(g * n_dataset_g, (g + 1) * n_dataset_g)),
                    ),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=False if device == "cpu" else True,
                )

                iterator = iter(train_dataloader)
                for _ in range(len(train_dataloader) - int(train_dataloader.drop_last)):
                    x = next(iterator)
                    x = x[0] if isinstance(x, list) or isinstance(x, tuple) else x
                    x = x.to(device)

                    # calculate batch size
                    bsize = x.size()[0]
                    if bsize + index > n_train_g:
                        bsize = n_train_g - index

                    if bsize == 0:
                        continue

                    # choose operator and generate measurement
                    y, params = measure(x, b=bsize, g=g)

                    # Add new data to it
                    if isinstance(y, TensorList):
                        for i in range(len(y)):
                            hf[f"y{i}_train"][index : index + bsize] = (
                                y[i][:bsize, :].to("cpu").numpy()
                            )
                    else:
                        hf["y_train"][index : index + bsize] = (
                            y[:bsize, :].to("cpu").numpy()
                        )
                    if supervised:
                        hf["x_train"][index : index + bsize] = (
                            x[:bsize, ...].to("cpu").numpy()
                        )
                    if save_physics_generator_params:
                        for p in params.keys():
                            hf[f"{p}_train"][index : index + bsize] = (
                                params[p][:bsize, ...].to("cpu").numpy()
                            )
                    index = index + bsize

        if test_dataset is not None:
            index = 0
            test_dataloader = DataLoader(
                Subset(
                    test_dataset, indices=list(range(g * n_test_g, (g + 1) * n_test_g))
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            iterator = iter(test_dataloader)
            for i in range(len(test_dataloader) - int(test_dataloader.drop_last)):
                x = next(iterator)
                x = x[0] if isinstance(x, list) or isinstance(x, tuple) else x
                x = x.to(device)

                bsize = x.size()[0]

                # choose operator and generate measurement
                y, params = measure(x, b=bsize, g=g)

                if i == 0:
                    hf.create_dataset(
                        "x_test", (n_test_g,) + x.shape[1:], dtype=x.cpu().numpy().dtype
                    )
                    if isinstance(y, TensorList):
                        for i in range(len(y)):
                            hf.create_dataset(
                                f"y{i}_test",
                                (n_test_g,) + y[i].shape[1:],
                                dtype=y[0].cpu().numpy().dtype,
                            )
                    else:
                        hf.create_dataset(
                            "y_test",
                            (n_test_g,) + y.shape[1:],
                            dtype=y.cpu().numpy().dtype,
                        )
                    if save_physics_generator_params:
                        for k, p in params.items():
                            hf.create_dataset(
                                f"{k}_test",
                                (n_test_g,) + p.shape[1:],
                                dtype=p.cpu().numpy().dtype,
                            )

                if isinstance(y, TensorList):
                    for i in range(len(y)):
                        hf[f"y{i}_test"][index : index + bsize] = y[i].to("cpu").numpy()
                else:
                    hf["y_test"][index : index + bsize] = y.to("cpu").numpy()

                hf["x_test"][index : index + bsize] = x.to("cpu").numpy()

                if save_physics_generator_params:
                    for p in params.keys():
                        hf[f"{p}_test"][index : index + bsize] = (
                            params[p].to("cpu").numpy()
                        )
                index = index + bsize

        if val_dataset is not None:
            index = 0
            val_dataloader = DataLoader(
                Subset(
                    val_dataset, indices=list(range(g * n_val_g, (g + 1) * n_val_g))
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            iterator = iter(val_dataloader)
            for i in range(len(val_dataloader) - int(val_dataloader.drop_last)):
                x = next(iterator)
                x = x[0] if isinstance(x, list) or isinstance(x, tuple) else x
                x = x.to(device)

                bsize = x.size()[0]

                # choose operator and generate measurement
                y, params = measure(x, b=bsize, g=g)

                if i == 0:
                    hf.create_dataset(
                        "x_val", (n_val_g,) + x.shape[1:], dtype=x.cpu().numpy().dtype
                    )
                    hf.create_dataset(
                        "y_val", (n_val_g,) + y.shape[1:], dtype=y.cpu().numpy().dtype
                    )
                    if save_physics_generator_params:
                        for k, p in params.items():
                            hf.create_dataset(
                                f"{k}_val",
                                (n_val_g,) + p.shape[1:],
                                dtype=p.cpu().numpy().dtype,
                            )

                hf["x_val"][index : index + bsize] = x.to("cpu").numpy()
                hf["y_val"][index : index + bsize] = y.to("cpu").numpy()

                if save_physics_generator_params:
                    for p in params.keys():
                        hf[f"{p}_val"][index : index + bsize] = (
                            params[p].to("cpu").numpy()
                        )
                index = index + bsize

        hf.close()

        if verbose:
            print(f"Dataset has been saved at {hf_path}")

    return hf_paths[0] if G == 1 else hf_paths
